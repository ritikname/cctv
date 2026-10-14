"""
Retail Store CCTV Video Analytics Pipeline v13.2

Features:
- YOLO + BoT-SORT tracking
- OSNet MSMT17 ReID gallery
- Visitor-ID gallery with co-presence exclusivity
- Multi-person group detection
- Group requires configurable together time (default 5 seconds)
- likely_group if together for >= configurable fraction of full video (default 50%)
- Final annotated video shows G1, G2, etc.
"""

import json
import time
import warnings
import os
import ssl
import tempfile

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

warnings.filterwarnings("ignore")


# =====================================================================
# CONFIGURATION
# =====================================================================
@dataclass
class PipelineConfig:
    detector_model: str = "yolo11x.pt"
    detector_confidence: float = 0.40
    detector_person_class: int = 0
    detector_imgsz: int = 0  # 0 = Ultralytics default

    tracker_type: str = "botsort"

    # OSNet ReID
    reid_input_height: int = 256
    reid_input_width: int = 128
    reid_embedding_dim: int = 512
    reid_min_crop_height: int = 40
    reid_min_crop_width: int = 20

    # Gallery
    gallery_rolling_size: int = 10
    gallery_sample_rate_hz: float = 10.0
    gallery_min_crop_quality: float = 0.25

    # Identity Matching
    gallery_match_threshold: float = 0.90
    gallery_new_person_confirm_count: int = 3
    gallery_new_person_confirm_window_sec: float = 2.0
    max_physical_speed_px_frame: float = 25.0

    # Track Validation
    min_track_frames: int = 6

    # Group Detection
    group_distance_threshold_px: float = 140.0
    group_min_overlap_frames: int = 25
    group_proximity_ratio: float = 0.60

    # Required together duration before declaring a group
    group_min_together_seconds: float = 5.0

    # Allows tiny missing detection gaps while treating people as together
    group_max_gap_frames: int = 3

    # If together for >= this fraction of entire video => likely_group
    group_likely_video_fraction: float = 0.50

    # True = must be continuously together for group_min_together_seconds
    # False = accumulated close time is enough
    group_require_continuous: bool = True

    # Normally staff/cashiers are excluded from customer groups
    group_include_staff: bool = False

    # Staff/Cashier Detection
    staff_min_presence_ratio: float = 0.50
    staff_max_movement_radius_px: float = 85.0

    # Zones
    enable_zones: bool = True
    dwell_min_frames: int = 15

    # Entrance Line
    enable_entrance_line: bool = False
    entrance_cooldown_frames: int = 20

    # Preview / Debug
    show_preview: bool = True
    show_tracker_id: bool = False
    debug_identity_log: bool = True
    debug_log_path: str = "identity_debug.csv"
    similarity_stats_path: str = "similarity_stats.json"

    # Output
    output_video: bool = True
    output_json: bool = True
    output_json_path: str = "analytics_output.json"
    output_video_path: str = "annotated_output.mp4"

    fps: float = 30.0


# =====================================================================
# OSNET MSMT17
# =====================================================================
OSNET_REID_REPO = "kaiyangzhou/osnet"

OSNET_REID_FILENAME = (
    "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_"
    "stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
)


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        k,
        stride=1,
        padding=0,
        groups=1,
        use_relu=True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_c,
            out_c,
            k,
            stride,
            padding,
            groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True) if use_relu else None

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x) if self.relu else x


class Conv1x1(ConvLayer):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__(in_c, out_c, 1, stride, 0, groups)


class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(
            out_c,
            out_c,
            3,
            padding=1,
            groups=out_c,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class ChannelGate(nn.Module):
    def __init__(self, in_c, num_gates=None, reduction=16):
        super().__init__()

        num_gates = num_gates or in_c

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, num_gates, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        return x * y


class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction=4):
        super().__init__()

        mid = out_c // reduction

        self.conv1 = Conv1x1(in_c, mid)

        self.conv2a = LightConv3x3(mid, mid)

        self.conv2b = nn.Sequential(
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
        )

        self.conv2c = nn.Sequential(
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
        )

        self.conv2d = nn.Sequential(
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
            LightConv3x3(mid, mid),
        )

        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1(mid, out_c)
        self.downsample = Conv1x1(in_c, out_c) if in_c != out_c else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x1 = self.conv1(x)

        x2 = (
            self.gate(self.conv2a(x1))
            + self.gate(self.conv2b(x1))
            + self.gate(self.conv2c(x1))
            + self.gate(self.conv2d(x1))
        )

        x3 = self.conv3(x2)

        if self.downsample is not None:
            identity = self.downsample(identity)

        return self.relu(x3 + identity)


class OSNet(nn.Module):
    def __init__(
        self,
        num_classes=1000,
        blocks=[2, 2, 2],
        channels=[64, 256, 384, 512],
        feature_dim=512,
    ):
        super().__init__()

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv2 = self._make_layer(
            blocks[0],
            channels[0],
            channels[1],
        )

        self.pool2 = nn.Sequential(
            Conv1x1(channels[1], channels[1]),
            nn.AvgPool2d(2, stride=2),
        )

        self.conv3 = self._make_layer(
            blocks[1],
            channels[1],
            channels[2],
        )

        self.pool3 = nn.Sequential(
            Conv1x1(channels[2], channels[2]),
            nn.AvgPool2d(2, stride=2),
        )

        self.conv4 = self._make_layer(
            blocks[2],
            channels[2],
            channels[3],
        )

        self.conv5 = Conv1x1(channels[3], channels[3])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[3], feature_dim)
        self.bn = nn.BatchNorm1d(feature_dim)
        self.feature_dim = feature_dim

    def _make_layer(self, n_blocks, in_c, out_c):
        layers = [OSBlock(in_c, out_c)]

        for _ in range(n_blocks - 1):
            layers.append(OSBlock(out_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.conv5(self.conv4(x))
        x = self.gap(x).flatten(1)
        x = self.bn(self.fc(x))
        return F.normalize(x, p=2, dim=1)


class ReIDExtractor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = None
        self.backend = None

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((
                config.reid_input_height,
                config.reid_input_width,
            )),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._load_model()

    def _load_model(self):
        ssl._create_default_https_context = ssl._create_unverified_context

        try:
            print("[-] Downloading/loading MSMT17 OSNet weights...")

            weight_path = hf_hub_download(
                repo_id=OSNET_REID_REPO,
                filename=OSNET_REID_FILENAME,
                local_dir=".",
                local_dir_use_symlinks=False,
            )

            model = OSNet(feature_dim=512)

            state_dict = torch.load(
                weight_path,
                map_location=self.device,
            )

            model_dict = model.state_dict()
            matched = {}

            for k, v in state_dict.items():
                k_clean = k.replace("module.", "")

                if k_clean in model_dict and v.shape == model_dict[k_clean].shape:
                    matched[k_clean] = v

            model_dict.update(matched)

            model.load_state_dict(
                model_dict,
                strict=False,
            )

            model.to(self.device).eval()

            self.model = model
            self.backend = "osnet_msmt17"

            print(
                f"[+] ReID: OSNet MSMT17 loaded "
                f"({len(matched)} layers matched)"
            )

        except Exception as e:
            print(f"[!] OSNet failed: {e}")
            print("[!] Falling back to ResNet-18.")

            import torchvision.models as tv_models

            model = tv_models.resnet18(
                weights=tv_models.ResNet18_Weights.DEFAULT
            )

            model.fc = nn.Identity()
            model.to(self.device).eval()

            self.model = model
            self.backend = "resnet18"

    @torch.no_grad()
    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty(
                (0, self.config.reid_embedding_dim),
                dtype=np.float32,
            )

        tensors = []

        for crop in crops:
            if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
                crop = np.zeros(
                    (
                        self.config.reid_input_height,
                        self.config.reid_input_width,
                        3,
                    ),
                    dtype=np.uint8,
                )

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(rgb))

        batch = torch.stack(tensors).to(self.device)

        feats = self.model(batch)

        if isinstance(feats, torch.Tensor):
            feats = feats.cpu().numpy()

        feats = np.asarray(feats, dtype=np.float32)

        return feats / (
            np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
        )

    def compute_quality(self, cw, ch, x1, y1, x2, y2, fw, fh):
        if (
            ch < self.config.reid_min_crop_height
            or cw < self.config.reid_min_crop_width
        ):
            return 0.15

        resolution_score = min(
            1.0,
            (ch * cw)
            / (
                self.config.reid_min_crop_height
                * self.config.reid_min_crop_width
                * 5
            ),
        )

        aspect = cw / max(ch, 1)

        aspect_score = 1.0 - min(
            1.0,
            abs(aspect - 0.45) / 0.45,
        )

        margin = 15

        border_score = (
            0.7
            if (
                x1 < margin
                or y1 < margin
                or x2 > fw - margin
                or y2 > fh - margin
            )
            else 1.0
        )

        return float(
            np.clip(
                resolution_score * aspect_score * border_score,
                0.15,
                1.0,
            )
        )


# =====================================================================
# IDENTITY GALLERY
# =====================================================================
@dataclass
class GalleryPerson:
    visitor_id: int
    embeddings: deque = field(default_factory=lambda: deque(maxlen=10))
    qualities: deque = field(default_factory=lambda: deque(maxlen=10))
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_position: Optional[Tuple[float, float]] = None
    is_active_in_frame: bool = False
    locked_to_track_id: Optional[int] = None


@dataclass
class CandidatePerson:
    track_id: int
    embeddings: List[np.ndarray] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    last_position: Optional[Tuple[float, float]] = None


class RollingGallery:
    def __init__(self, config: PipelineConfig):
        self.config = config

        self.visitors: Dict[int, GalleryPerson] = {}
        self.candidates: Dict[int, CandidatePerson] = {}

        self.track_to_visitor: Dict[int, int] = {}
        self.last_sample_frame_per_track: Dict[int, int] = {}

        self.decisions_log: List[Dict] = []
        self.match_similarities: List[float] = []
        self.nonmatch_similarities: List[float] = []

        self.next_visitor_id = 1

    def should_sample(self, track_id: int, frame_idx: int) -> bool:
        interval = max(
            1,
            int(
                round(
                    self.config.fps
                    / self.config.gallery_sample_rate_hz
                )
            ),
        )

        last = self.last_sample_frame_per_track.get(
            track_id,
            -999999,
        )

        return (frame_idx - last) >= interval

    def _max_similarity(
        self,
        embedding: np.ndarray,
        person_embeddings: List[np.ndarray],
    ) -> float:
        if not person_embeddings:
            return 0.0

        gallery = np.asarray(person_embeddings, dtype=np.float32)
        similarities = gallery @ embedding

        return float(np.max(similarities))

    def _check_speed_valid(
        self,
        person: GalleryPerson,
        current_pos: Tuple[float, float],
        frame_idx: int,
    ) -> bool:
        if person.last_position is None or current_pos is None:
            return True

        frame_gap = max(
            1,
            frame_idx - person.last_seen_frame,
        )

        distance = np.sqrt(
            (person.last_position[0] - current_pos[0]) ** 2
            + (person.last_position[1] - current_pos[1]) ** 2
        )

        speed = distance / frame_gap

        return speed <= self.config.max_physical_speed_px_frame

    def query(
        self,
        track_id: int,
        embedding: np.ndarray,
        quality: float,
        frame_idx: int,
        position: Tuple[float, float],
        currently_locked_vids: Set[int],
    ) -> Tuple[Optional[int], str, float]:

        # Existing BoT-SORT tracker ID already mapped to a visitor ID.
        if track_id in self.track_to_visitor:
            vid = self.track_to_visitor[track_id]

            if vid in self.visitors:
                person = self.visitors[vid]

                if quality >= self.config.gallery_min_crop_quality:
                    person.embeddings.append(embedding.copy())
                    person.qualities.append(quality)

                person.last_seen_frame = frame_idx
                person.last_position = position
                person.is_active_in_frame = True
                person.locked_to_track_id = track_id

                self.last_sample_frame_per_track[track_id] = frame_idx

                return vid, "continuous_update", 1.0

        # Crop is not useful enough for ReID.
        if quality < self.config.gallery_min_crop_quality:
            return None, "rejected_low_quality", 0.0

        best_vid = None
        best_similarity = -1.0
        max_evaluated_similarity = 0.0

        for vid, person in self.visitors.items():
            # Do not let 2 visible people take the same visitor ID.
            if vid in currently_locked_vids:
                continue

            similarity = self._max_similarity(
                embedding,
                list(person.embeddings),
            )

            max_evaluated_similarity = max(
                max_evaluated_similarity,
                similarity,
            )

            if similarity < self.config.gallery_match_threshold:
                continue

            if not self._check_speed_valid(
                person,
                position,
                frame_idx,
            ):
                continue

            if similarity > best_similarity:
                best_similarity = similarity
                best_vid = vid

        # Match this new tracker ID to an existing visitor ID.
        if best_vid is not None:
            person = self.visitors[best_vid]

            person.embeddings.append(embedding.copy())
            person.qualities.append(quality)
            person.last_seen_frame = frame_idx
            person.last_position = position
            person.is_active_in_frame = True
            person.locked_to_track_id = track_id

            self.track_to_visitor[track_id] = best_vid
            self.last_sample_frame_per_track[track_id] = frame_idx

            self.match_similarities.append(best_similarity)

            self._log(
                frame_idx,
                track_id,
                best_vid,
                "matched_existing",
                best_similarity,
            )

            return best_vid, "matched_existing", best_similarity

        # No match: put track into pending candidate pool.
        self.nonmatch_similarities.append(max_evaluated_similarity)

        candidate = self.candidates.get(track_id)

        if candidate is None:
            candidate = CandidatePerson(
                track_id=track_id,
                embeddings=[embedding.copy()],
                qualities=[quality],
                first_frame=frame_idx,
                last_frame=frame_idx,
                last_position=position,
            )

            self.candidates[track_id] = candidate
            self.last_sample_frame_per_track[track_id] = frame_idx

            self._log(
                frame_idx,
                track_id,
                None,
                "candidate_pending",
                max_evaluated_similarity,
            )

            return None, "candidate_pending", max_evaluated_similarity

        candidate.embeddings.append(embedding.copy())
        candidate.qualities.append(quality)
        candidate.last_frame = frame_idx
        candidate.last_position = position

        self.last_sample_frame_per_track[track_id] = frame_idx

        window_frames = int(
            self.config.gallery_new_person_confirm_window_sec
            * self.config.fps
        )

        if frame_idx - candidate.first_frame > window_frames:
            candidate.embeddings = candidate.embeddings[-1:]
            candidate.qualities = candidate.qualities[-1:]
            candidate.first_frame = frame_idx

        if len(candidate.embeddings) >= self.config.gallery_new_person_confirm_count:
            vid = self.next_visitor_id
            self.next_visitor_id += 1

            person = GalleryPerson(
                visitor_id=vid,
                embeddings=deque(
                    candidate.embeddings,
                    maxlen=self.config.gallery_rolling_size,
                ),
                qualities=deque(
                    candidate.qualities,
                    maxlen=self.config.gallery_rolling_size,
                ),
                first_seen_frame=candidate.first_frame,
                last_seen_frame=frame_idx,
                last_position=position,
                is_active_in_frame=True,
                locked_to_track_id=track_id,
            )

            self.visitors[vid] = person
            self.track_to_visitor[track_id] = vid

            self.candidates.pop(track_id, None)

            self._log(
                frame_idx,
                track_id,
                vid,
                "confirmed_new",
                max_evaluated_similarity,
            )

            return vid, "confirmed_new", max_evaluated_similarity

        self._log(
            frame_idx,
            track_id,
            None,
            "candidate_pending",
            max_evaluated_similarity,
        )

        return None, "candidate_pending", max_evaluated_similarity

    def reset_frame_locks(self):
        for person in self.visitors.values():
            person.is_active_in_frame = False
            person.locked_to_track_id = None

    def retire_dead_track(self, track_id: int):
        self.candidates.pop(track_id, None)
        self.last_sample_frame_per_track.pop(track_id, None)

    def distinct_count(self) -> int:
        return len(self.visitors)

    def _log(self, frame, track_id, visitor_id, decision, similarity):
        self.decisions_log.append({
            "frame": frame,
            "track_id": track_id,
            "visitor_id": visitor_id,
            "decision": decision,
            "similarity": round(float(similarity), 4),
        })

    def dump_similarity_stats(self, path: str):
        def stats(values):
            if not values:
                return {"count": 0}

            arr = np.asarray(values, dtype=np.float32)

            return {
                "count": int(len(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }

        output = {
            "match_similarities": stats(self.match_similarities),
            "nonmatch_similarities": stats(self.nonmatch_similarities),
            "current_threshold": self.config.gallery_match_threshold,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)


# =====================================================================
# TRAJECTORIES
# =====================================================================
@dataclass
class VisitorTrajectory:
    visitor_id: int
    positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    first_frame: int = 999999999
    last_frame: int = 0


class TrajectoryStore:
    def __init__(self):
        self.trajectories: Dict[int, VisitorTrajectory] = {}

    def update(
        self,
        visitor_id: int,
        frame_idx: int,
        bbox: Tuple[int, int, int, int],
    ):
        if visitor_id not in self.trajectories:
            self.trajectories[visitor_id] = VisitorTrajectory(
                visitor_id=visitor_id
            )

        trajectory = self.trajectories[visitor_id]

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        trajectory.positions[frame_idx] = (cx, cy)
        trajectory.bboxes[frame_idx] = bbox

        trajectory.first_frame = min(
            trajectory.first_frame,
            frame_idx,
        )

        trajectory.last_frame = max(
            trajectory.last_frame,
            frame_idx,
        )


# =====================================================================
# GROUP DETECTION
# =====================================================================
@dataclass
class Group:
    group_id: int
    members: List[int]
    size: int
    status: str = "group"
    evidence: Dict = field(default_factory=dict)

    # Used only internally for final video rendering.
    active_frames: Set[int] = field(default_factory=set, repr=False)


class AnalyticsEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def classify_staff(
        self,
        trajectories: Dict[int, VisitorTrajectory],
        total_frames: int,
    ) -> Dict[int, str]:

        roles = {}

        for visitor_id, trajectory in trajectories.items():
            frames_seen = list(trajectory.positions.keys())

            if len(frames_seen) < self.config.min_track_frames:
                roles[visitor_id] = "Customer"
                continue

            presence_ratio = len(frames_seen) / max(total_frames, 1)

            positions = np.asarray(
                [trajectory.positions[f] for f in frames_seen],
                dtype=np.float32,
            )

            if len(positions) > 10:
                center = np.mean(positions, axis=0)

                max_radius = float(
                    np.max(
                        np.sqrt(
                            np.sum(
                                (positions - center) ** 2,
                                axis=1,
                            )
                        )
                    )
                )
            else:
                max_radius = 999.0

            if (
                presence_ratio >= self.config.staff_min_presence_ratio
                and max_radius <= self.config.staff_max_movement_radius_px
            ):
                roles[visitor_id] = "Staff/Cashier"
            else:
                roles[visitor_id] = "Customer"

        return roles

    def _compute_together_metrics(
        self,
        member_ids: List[int],
        trajectories: Dict[int, VisitorTrajectory],
        total_frames: int,
    ) -> Tuple[Optional[Dict], bool, Set[int]]:
        """
        For all members of a candidate group:
        - all must exist in same frame
        - all pair distances must be <= group_distance_threshold_px
        """

        member_ids = sorted(member_ids)

        if len(member_ids) < 2:
            return None, False, set()

        frame_sets = [
            set(trajectories[vid].positions.keys())
            for vid in member_ids
        ]

        if not frame_sets:
            return None, False, set()

        common_frames = sorted(set.intersection(*frame_sets))

        if len(common_frames) < self.config.group_min_overlap_frames:
            return None, False, set()

        close_frames = []
        frame_distances = []

        current_run = []
        close_runs = []

        last_close_frame = None
        max_gap = self.config.group_max_gap_frames + 1

        for frame_idx in common_frames:
            all_pair_distances = []

            for va, vb in combinations(member_ids, 2):
                ax, ay = trajectories[va].positions[frame_idx]
                bx, by = trajectories[vb].positions[frame_idx]

                dist = float(
                    np.sqrt(
                        (ax - bx) ** 2
                        + (ay - by) ** 2
                    )
                )

                all_pair_distances.append(dist)

            max_pair_distance = max(all_pair_distances)
            avg_pair_distance = float(np.mean(all_pair_distances))

            frame_distances.append(avg_pair_distance)

            is_close = (
                max_pair_distance
                <= self.config.group_distance_threshold_px
            )

            if is_close:
                close_frames.append(frame_idx)

                if not current_run:
                    current_run = [frame_idx]

                elif frame_idx - last_close_frame <= max_gap:
                    current_run.append(frame_idx)

                else:
                    close_runs.append(current_run)
                    current_run = [frame_idx]

                last_close_frame = frame_idx

            else:
                if current_run:
                    close_runs.append(current_run)
                    current_run = []

                last_close_frame = None

        if current_run:
            close_runs.append(current_run)

        fps = max(self.config.fps, 1.0)

        total_close_frames = len(close_frames)
        shared_frames = len(common_frames)

        total_close_seconds = total_close_frames / fps

        longest_continuous_frames = 0
        qualifying_run_frames: Set[int] = set()

        for run in close_runs:
            if not run:
                continue

            run_duration_frames = run[-1] - run[0] + 1

            longest_continuous_frames = max(
                longest_continuous_frames,
                run_duration_frames,
            )

            run_seconds = run_duration_frames / fps

            if run_seconds >= self.config.group_min_together_seconds:
                qualifying_run_frames.update(run)

        longest_continuous_seconds = (
            longest_continuous_frames / fps
        )

        close_ratio_shared = (
            total_close_frames / max(shared_frames, 1)
        )

        video_together_ratio = (
            total_close_frames / max(total_frames, 1)
        )

        if self.config.group_require_continuous:
            time_requirement_value = longest_continuous_seconds
            display_frames = qualifying_run_frames
        else:
            time_requirement_value = total_close_seconds
            display_frames = set(close_frames)

        qualifies = (
            time_requirement_value
            >= self.config.group_min_together_seconds
            and close_ratio_shared
            >= self.config.group_proximity_ratio
        )

        if not qualifies:
            display_frames = set()

        avg_distance = (
            float(np.mean(frame_distances))
            if frame_distances
            else 9999.0
        )

        distance_score = max(
            0.0,
            1.0
            - avg_distance
            / max(
                self.config.group_distance_threshold_px * 2.0,
                1.0,
            ),
        )

        duration_score = min(
            1.0,
            time_requirement_value
            / max(
                self.config.group_min_together_seconds,
                0.001,
            ),
        )

        bond_score = (
            0.50 * close_ratio_shared
            + 0.30 * duration_score
            + 0.20 * distance_score
        )

        evidence = {
            "members": [f"V{vid}" for vid in member_ids],
            "member_count": len(member_ids),
            "shared_frames": shared_frames,
            "close_frames": total_close_frames,
            "avg_distance_px": round(avg_distance, 1),
            "close_ratio_shared": round(close_ratio_shared, 3),
            "total_close_seconds": round(total_close_seconds, 2),
            "longest_continuous_close_seconds": round(
                longest_continuous_seconds,
                2,
            ),
            "video_together_ratio": round(video_together_ratio, 3),
            "bond_score": round(bond_score, 3),
            "continuous_required": self.config.group_require_continuous,
            "qualifies_as_group": qualifies,
        }

        return evidence, qualifies, display_frames

    def detect_groups(
        self,
        trajectories: Dict[int, VisitorTrajectory],
        roles: Dict[int, str],
        total_frames: int,
    ) -> Tuple[List[Group], Dict]:

        if self.config.group_include_staff:
            eligible_visitors = sorted(trajectories.keys())
        else:
            eligible_visitors = sorted([
                visitor_id
                for visitor_id in trajectories.keys()
                if roles.get(visitor_id, "Customer") == "Customer"
            ])

        pair_evidence = {}
        valid_pairs = {}

        # Evaluate every pair.
        for va, vb in combinations(eligible_visitors, 2):
            evidence, qualifies, _ = self._compute_together_metrics(
                [va, vb],
                trajectories,
                total_frames,
            )

            if evidence is None:
                continue

            pair_evidence[(va, vb)] = evidence

            if qualifies:
                valid_pairs[(va, vb)] = evidence

        def pair_is_valid(va: int, vb: int) -> bool:
            return tuple(sorted((va, vb))) in valid_pairs

        def all_pairs_valid(members: Set[int]) -> bool:
            for va, vb in combinations(sorted(members), 2):
                if not pair_is_valid(va, vb):
                    return False
            return True

        def get_cluster_index(
            clusters: List[Set[int]],
            visitor_id: int,
        ) -> Optional[int]:
            for index, cluster in enumerate(clusters):
                if visitor_id in cluster:
                    return index
            return None

        def can_form_group(
            members: Set[int],
        ) -> Tuple[bool, Optional[Dict], Set[int]]:
            if not all_pairs_valid(members):
                return False, None, set()

            evidence, qualifies, active_frames = (
                self._compute_together_metrics(
                    sorted(members),
                    trajectories,
                    total_frames,
                )
            )

            return qualifies, evidence, active_frames

        # Exclusive greedy clustering.
        clusters: List[Set[int]] = []

        ranked_pairs = sorted(
            valid_pairs.items(),
            key=lambda item: item[1]["bond_score"],
            reverse=True,
        )

        for (va, vb), _ in ranked_pairs:
            cluster_a = get_cluster_index(clusters, va)
            cluster_b = get_cluster_index(clusters, vb)

            # Both are not part of groups yet.
            if cluster_a is None and cluster_b is None:
                candidate = {va, vb}

                allowed, _, _ = can_form_group(candidate)

                if allowed:
                    clusters.append(candidate)

                continue

            # Both are already in same group.
            if cluster_a is not None and cluster_a == cluster_b:
                continue

            # A is in a group, B is not.
            if cluster_a is not None and cluster_b is None:
                candidate = set(clusters[cluster_a])
                candidate.add(vb)

                allowed, _, _ = can_form_group(candidate)

                if allowed:
                    clusters[cluster_a] = candidate

                continue

            # B is in a group, A is not.
            if cluster_b is not None and cluster_a is None:
                candidate = set(clusters[cluster_b])
                candidate.add(va)

                allowed, _, _ = can_form_group(candidate)

                if allowed:
                    clusters[cluster_b] = candidate

                continue

            # Both belong to different clusters: merge only if all members qualify.
            candidate = set(clusters[cluster_a]) | set(clusters[cluster_b])

            allowed, _, _ = can_form_group(candidate)

            if allowed:
                for index in sorted(
                    [cluster_a, cluster_b],
                    reverse=True,
                ):
                    clusters.pop(index)

                clusters.append(candidate)

        groups = []

        for members_set in clusters:
            members = sorted(members_set)

            evidence, qualifies, active_frames = (
                self._compute_together_metrics(
                    members,
                    trajectories,
                    total_frames,
                )
            )

            if not qualifies or evidence is None:
                continue

            likely_group = (
                evidence["video_together_ratio"]
                >= self.config.group_likely_video_fraction
            )

            status = (
                "likely_group"
                if likely_group
                else "group"
            )

            evidence["classification"] = status
            evidence["likely_group_fraction_threshold"] = (
                self.config.group_likely_video_fraction
            )

            groups.append(
                Group(
                    group_id=0,
                    members=members,
                    size=len(members),
                    status=status,
                    evidence=evidence,
                    active_frames=active_frames,
                )
            )

        # Stable Group 1 / Group 2 ordering.
        groups.sort(
            key=lambda group: (
                min(
                    trajectories[vid].first_frame
                    for vid in group.members
                ),
                group.members,
            )
        )

        for index, group in enumerate(groups, start=1):
            group.group_id = index

        return groups, pair_evidence


# =====================================================================
# ZONES
# =====================================================================
@dataclass
class AnalyticsZone:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    color: Tuple[int, int, int] = (0, 255, 255)


class ZoneDwellEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.zones: List[AnalyticsZone] = []

        self.visitor_zone_frames = defaultdict(
            lambda: defaultdict(int)
        )

    def init_default(self, width: int, height: int):
        if self.zones:
            return

        w3 = width // 3
        h2 = height // 2

        self.zones = [
            AnalyticsZone(
                "zone_left",
                0,
                h2,
                w3,
                height,
                (255, 200, 0),
            ),
            AnalyticsZone(
                "zone_center",
                w3,
                h2,
                2 * w3,
                height,
                (0, 255, 200),
            ),
            AnalyticsZone(
                "zone_right",
                2 * w3,
                h2,
                width,
                height,
                (200, 0, 255),
            ),
        ]

    def update(
        self,
        visitor_id: int,
        bbox: Tuple[int, int, int, int],
    ):
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]

        for zone in self.zones:
            if (
                zone.x1 <= foot_x <= zone.x2
                and zone.y1 <= foot_y <= zone.y2
            ):
                self.visitor_zone_frames[visitor_id][zone.name] += 1

    def get_dwell(
        self,
        visitor_id: int,
        fps: float,
    ) -> Dict[str, float]:
        return {
            zone_name: round(frame_count / max(fps, 1), 2)
            for zone_name, frame_count
            in self.visitor_zone_frames.get(visitor_id, {}).items()
            if frame_count >= self.config.dwell_min_frames
        }


# =====================================================================
# ENTRANCE
# =====================================================================
@dataclass
class EntranceZone:
    name: str
    point_a: Tuple[int, int]
    point_b: Tuple[int, int]
    inside_direction: str = "below"


class EntranceEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config

        self.zones: List[EntranceZone] = []

        self.entry_count = 0
        self.exit_count = 0

        self.events: List[Dict] = []

        self.prev_side: Dict[int, int] = {}
        self.cooldown = defaultdict(int)

        self.velocity_buffer = defaultdict(
            lambda: deque(maxlen=5)
        )

    def init_default(self, width: int, height: int):
        if self.zones or not self.config.enable_entrance_line:
            return

        y = int(height * 0.50)

        self.zones.append(
            EntranceZone(
                "main_entrance",
                (0, y),
                (width, y),
                "below",
            )
        )

    def _side(
        self,
        point: Tuple[float, float],
        zone: EntranceZone,
    ) -> int:
        ax, ay = zone.point_a
        bx, by = zone.point_b
        px, py = point

        cross = (
            (bx - ax) * (py - ay)
            - (by - ay) * (px - ax)
        )

        return 1 if cross > 0 else -1

    def check(
        self,
        visitor_id: int,
        bbox: Tuple[int, int, int, int],
        frame_idx: int,
        timestamp: float,
        prev_bbox: Optional[Tuple[int, int, int, int]] = None,
    ):
        if (
            not self.config.enable_entrance_line
            or not self.zones
        ):
            return None

        if self.cooldown[visitor_id] > 0:
            self.cooldown[visitor_id] -= 1
            return None

        foot = (
            (bbox[0] + bbox[2]) / 2,
            bbox[3],
        )

        if prev_bbox is not None:
            prev_foot = (
                (prev_bbox[0] + prev_bbox[2]) / 2,
                prev_bbox[3],
            )

            self.velocity_buffer[visitor_id].append(
                foot[1] - prev_foot[1]
            )

        for zone in self.zones:
            current_side = self._side(foot, zone)
            previous_side = self.prev_side.get(visitor_id)

            if previous_side is None:
                self.prev_side[visitor_id] = current_side
                return None

            if current_side != previous_side:
                if len(self.velocity_buffer[visitor_id]) < 2:
                    self.prev_side[visitor_id] = current_side
                    return None

                avg_velocity_y = float(
                    np.mean(
                        list(self.velocity_buffer[visitor_id])
                    )
                )

                if abs(avg_velocity_y) < 2.0:
                    self.prev_side[visitor_id] = current_side
                    return None

                if zone.inside_direction == "below":
                    direction = (
                        "entry"
                        if avg_velocity_y > 0
                        else "exit"
                    )
                else:
                    direction = (
                        "entry"
                        if avg_velocity_y < 0
                        else "exit"
                    )

                self.prev_side[visitor_id] = current_side
                self.cooldown[visitor_id] = (
                    self.config.entrance_cooldown_frames
                )

                if direction == "entry":
                    self.entry_count += 1
                else:
                    self.exit_count += 1

                event = {
                    "type": direction,
                    "zone": zone.name,
                    "visitor_id": visitor_id,
                    "frame": frame_idx,
                    "timestamp": timestamp,
                }

                self.events.append(event)

                return event

            self.prev_side[visitor_id] = current_side

        return None


# =====================================================================
# MAIN PIPELINE
# =====================================================================
class RetailAnalyticsPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        print("=" * 62)
        print("  RETAIL CCTV ANALYTICS PIPELINE v13.2")
        print("=" * 62)

        print("[-] Loading YOLO detector...")
        self.detector = YOLO(self.config.detector_model)

        self.tracker_config_path = self._generate_tracker_config()

        print(
            f"[-] Tracker: {self.config.tracker_type.upper()} "
            f"| GMC disabled | ReID fusion ON"
        )

        print("[-] Loading OSNet ReID Extractor...")
        self.reid = ReIDExtractor(self.config)

        self.gallery = RollingGallery(self.config)
        self.trajectories = TrajectoryStore()
        self.zone_engine = ZoneDwellEngine(self.config)
        self.entrance_engine = EntranceEngine(self.config)
        self.analytics_engine = AnalyticsEngine(self.config)

        self.frame_idx = 0
        self.start_time = time.time()

        self.zones_initialized = False

        self.previous_bboxes: Dict[
            int,
            Tuple[int, int, int, int],
        ] = {}

        self.active_track_ids_history: Set[int] = set()

        self.occupancy_timeline = []
        self.peak_occupancy = 0
        self.debug_log = []

        # Stores bbox + assigned visitor ID for final render pass.
        self.frame_records: List[
            Dict[int, Tuple[Tuple[int, int, int, int], Optional[int]]]
        ] = []

        self.final_groups: List[Group] = []
        self.final_roles: Dict[int, str] = {}

        print("[+] Ready.\n")

    def _generate_tracker_config(self) -> str:
        temp_dir = tempfile.gettempdir()

        path = os.path.join(
            temp_dir,
            f"custom_{self.config.tracker_type}_v132.yaml",
        )

        if self.config.tracker_type == "botsort":
            content = """tracker_type: botsort
track_high_thresh: 0.40
track_low_thresh: 0.10
new_track_thresh: 0.45
track_buffer: 90
match_thresh: 0.75
fuse_score: True
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: True
"""
        else:
            content = """tracker_type: bytetrack
track_high_thresh: 0.40
track_low_thresh: 0.10
new_track_thresh: 0.45
track_buffer: 90
match_thresh: 0.75
fuse_score: True
gmc_method: none
"""

        with open(path, "w") as f:
            f.write(content)

        return path

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]

        if not self.zones_initialized:
            if self.config.enable_zones:
                self.zone_engine.init_default(width, height)

            self.entrance_engine.init_default(width, height)

            self.zones_initialized = True

        timestamp = self.frame_idx / max(self.config.fps, 1.0)

        track_kwargs = {
            "persist": True,
            "conf": self.config.detector_confidence,
            "classes": [self.config.detector_person_class],
            "tracker": self.tracker_config_path,
            "verbose": False,
        }

        if self.config.detector_imgsz > 0:
            track_kwargs["imgsz"] = self.config.detector_imgsz

        results = self.detector.track(
            frame,
            **track_kwargs,
        )[0]

        active_track_bboxes: Dict[
            int,
            Tuple[int, int, int, int],
        ] = {}

        if (
            results.boxes is not None
            and results.boxes.id is not None
        ):
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.int().cpu().tolist()

            for bbox, track_id in zip(boxes, track_ids):
                active_track_bboxes[track_id] = tuple(bbox)

        # Retire candidates for fully missing tracker IDs.
        current_active_ids = set(active_track_bboxes.keys())

        for dead_track_id in (
            self.active_track_ids_history - current_active_ids
        ):
            self.gallery.retire_dead_track(dead_track_id)

        self.active_track_ids_history = current_active_ids

        # Reset locks at start of frame.
        self.gallery.reset_frame_locks()

        # Lock all visitor IDs already attached to active tracker IDs.
        # This prevents a newly created track from claiming an ID already
        # visible in the same frame.
        currently_locked_vids: Set[int] = set()

        for track_id, bbox in active_track_bboxes.items():
            visitor_id = self.gallery.track_to_visitor.get(track_id)

            if visitor_id is None:
                continue

            person = self.gallery.visitors.get(visitor_id)

            if person is None:
                continue

            person.is_active_in_frame = True
            person.locked_to_track_id = track_id
            person.last_seen_frame = self.frame_idx

            person.last_position = (
                (bbox[0] + bbox[2]) / 2.0,
                (bbox[1] + bbox[3]) / 2.0,
            )

            currently_locked_vids.add(visitor_id)

        # Prepare ReID crops.
        tids_to_sample = []
        crops_to_sample = []
        qualities_to_sample = []
        positions_to_sample = []

        for track_id, bbox in active_track_bboxes.items():
            if not self.gallery.should_sample(
                track_id,
                self.frame_idx,
            ):
                continue

            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])

            x2 = min(width, bbox[2])
            y2 = min(height, bbox[3])

            crop_width = x2 - x1
            crop_height = y2 - y1

            quality = self.reid.compute_quality(
                crop_width,
                crop_height,
                x1,
                y1,
                x2,
                y2,
                width,
                height,
            )

            if crop_height >= 10 and crop_width >= 5:
                crop = frame[y1:y2, x1:x2]
            else:
                crop = np.zeros(
                    (
                        self.config.reid_input_height,
                        self.config.reid_input_width,
                        3,
                    ),
                    dtype=np.uint8,
                )

            tids_to_sample.append(track_id)
            crops_to_sample.append(crop)
            qualities_to_sample.append(quality)

            positions_to_sample.append((
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            ))

        if crops_to_sample:
            embeddings = self.reid.extract(crops_to_sample)
        else:
            embeddings = np.empty(
                (0, self.config.reid_embedding_dim),
                dtype=np.float32,
            )

        # ReID gallery query.
        for track_id, embedding, quality, position in zip(
            tids_to_sample,
            embeddings,
            qualities_to_sample,
            positions_to_sample,
        ):
            visitor_id, decision, similarity = self.gallery.query(
                track_id,
                embedding,
                quality,
                self.frame_idx,
                position,
                currently_locked_vids,
            )

            if visitor_id is not None:
                currently_locked_vids.add(visitor_id)

            self.debug_log.append({
                "frame": self.frame_idx,
                "timestamp": round(timestamp, 2),
                "track_id": track_id,
                "visitor_id": visitor_id,
                "decision": decision,
                "similarity": round(float(similarity), 4),
                "quality": round(float(quality), 3),
            })

        # Update trajectory / zones / entrance.
        for track_id, bbox in active_track_bboxes.items():
            visitor_id = self.gallery.track_to_visitor.get(track_id)

            if visitor_id is None:
                continue

            self.trajectories.update(
                visitor_id,
                self.frame_idx,
                bbox,
            )

            if self.config.enable_zones:
                self.zone_engine.update(visitor_id, bbox)

            prev_bbox = self.previous_bboxes.get(visitor_id)

            self.entrance_engine.check(
                visitor_id,
                bbox,
                self.frame_idx,
                timestamp,
                prev_bbox,
            )

            self.previous_bboxes[visitor_id] = bbox

        current_occupancy = len(active_track_bboxes)

        self.peak_occupancy = max(
            self.peak_occupancy,
            current_occupancy,
        )

        if self.frame_idx % 25 == 0:
            self.occupancy_timeline.append({
                "frame": self.frame_idx,
                "timestamp": round(timestamp, 2),
                "occupancy": current_occupancy,
            })

        # Save detections and VID assignments for final group-video rendering.
        frame_record = {}

        for track_id, bbox in active_track_bboxes.items():
            frame_record[track_id] = (
                bbox,
                self.gallery.track_to_visitor.get(track_id),
            )

        self.frame_records.append(frame_record)

        annotated = self._draw_frame(
            frame=frame,
            frame_record=frame_record,
            frame_idx=self.frame_idx,
            current_occupancy=current_occupancy,
            group_by_visitor=None,
            final_render=False,
        )

        self.frame_idx += 1

        return annotated

    def _person_color(
        self,
        visitor_id: Optional[int],
        track_id: int,
    ) -> Tuple[int, int, int]:
        base = visitor_id if visitor_id is not None else track_id

        rng = np.random.default_rng(base * 17 + 41)

        return tuple(
            int(value)
            for value in rng.integers(70, 255, size=3)
        )

    def _draw_frame(
        self,
        frame: np.ndarray,
        frame_record: Dict[
            int,
            Tuple[Tuple[int, int, int, int], Optional[int]]
        ],
        frame_idx: int,
        current_occupancy: int,
        group_by_visitor: Optional[Dict[int, Group]],
        final_render: bool,
    ) -> np.ndarray:

        visual = frame.copy()

        # Draw zones.
        if self.config.enable_zones:
            for zone in self.zone_engine.zones:
                overlay = visual.copy()

                cv2.rectangle(
                    overlay,
                    (zone.x1, zone.y1),
                    (zone.x2, zone.y2),
                    zone.color,
                    -1,
                )

                cv2.addWeighted(
                    overlay,
                    0.12,
                    visual,
                    0.88,
                    0,
                    visual,
                )

                cv2.rectangle(
                    visual,
                    (zone.x1, zone.y1),
                    (zone.x2, zone.y2),
                    zone.color,
                    1,
                )

                cv2.putText(
                    visual,
                    zone.name,
                    (zone.x1 + 5, zone.y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    zone.color,
                    1,
                )

        # Draw entrance line.
        if self.config.enable_entrance_line:
            for entrance_zone in self.entrance_engine.zones:
                cv2.line(
                    visual,
                    entrance_zone.point_a,
                    entrance_zone.point_b,
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    visual,
                    f"ENTRANCE: {entrance_zone.name}",
                    (
                        entrance_zone.point_a[0] + 5,
                        entrance_zone.point_a[1] - 10,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        # Draw persons.
        for track_id, (bbox, visitor_id) in frame_record.items():
            color = self._person_color(visitor_id, track_id)

            x1, y1, x2, y2 = bbox

            cv2.rectangle(
                visual,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            if visitor_id is None:
                label = f"T{track_id} Pending"
            else:
                label = f"V{visitor_id}"

                if self.config.show_tracker_id:
                    label = f"T{track_id} | V{visitor_id}"

                # Show final G1/G2 label only during actual together interval.
                if (
                    final_render
                    and group_by_visitor is not None
                    and visitor_id in group_by_visitor
                ):
                    group = group_by_visitor[visitor_id]

                    if frame_idx in group.active_frames:
                        if group.status == "likely_group":
                            label += f" | G{group.group_id} Likely"
                        else:
                            label += f" | G{group.group_id}"

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            text_y1 = max(0, y1 - 24)

            cv2.rectangle(
                visual,
                (x1, text_y1),
                (x1 + text_size[0] + 8, y1),
                color,
                -1,
            )

            cv2.putText(
                visual,
                label,
                (x1 + 4, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Draw last 30 trajectory points up to current frame.
            if (
                visitor_id is not None
                and visitor_id in self.trajectories.trajectories
            ):
                trajectory = self.trajectories.trajectories[visitor_id]

                valid_frames = sorted([
                    f
                    for f in trajectory.positions.keys()
                    if f <= frame_idx
                ])[-30:]

                points = [
                    trajectory.positions[f]
                    for f in valid_frames
                ]

                for index in range(1, len(points)):
                    cv2.line(
                        visual,
                        (
                            int(points[index - 1][0]),
                            int(points[index - 1][1]),
                        ),
                        (
                            int(points[index][0]),
                            int(points[index][1]),
                        ),
                        color,
                        2,
                    )

        # HUD.
        if final_render:
            likely_count = sum(
                1
                for group in self.final_groups
                if group.status == "likely_group"
            )

            hud_items = [
                f"Frame: {frame_idx}",
                f"Occupancy: {current_occupancy} | Peak: {self.peak_occupancy}",
                f"Distinct Visitors: {self.gallery.distinct_count()}",
                f"Groups: {len(self.final_groups)} | Likely Groups: {likely_count}",
            ]
        else:
            hud_items = [
                f"Frame: {frame_idx}",
                f"Occupancy: {current_occupancy} | Peak: {self.peak_occupancy}",
                f"Distinct Visitors: {self.gallery.distinct_count()}",
                f"Candidates: {len(self.gallery.candidates)}",
                (
                    f"{self.config.tracker_type.upper()} + ReID "
                    f"| Threshold: {self.config.gallery_match_threshold}"
                ),
            ]

        if self.config.enable_entrance_line:
            hud_items.append(
                f"In: {self.entrance_engine.entry_count} | "
                f"Out: {self.entrance_engine.exit_count}"
            )

        hud_y = 30

        for item in hud_items:
            cv2.putText(
                visual,
                item,
                (10, hud_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

            hud_y += 24

        return visual

    def generate_report(self) -> Dict:
        roles = self.analytics_engine.classify_staff(
            self.trajectories.trajectories,
            self.frame_idx,
        )

        groups, pair_evidence = self.analytics_engine.detect_groups(
            self.trajectories.trajectories,
            roles,
            self.frame_idx,
        )

        self.final_roles = roles
        self.final_groups = groups

        customer_count = sum(
            1
            for role in roles.values()
            if role == "Customer"
        )

        staff_count = sum(
            1
            for role in roles.values()
            if role == "Staff/Cashier"
        )

        people = []

        for visitor_id, person in self.gallery.visitors.items():
            trajectory = self.trajectories.trajectories.get(visitor_id)

            if trajectory is None:
                continue

            duration = (
                trajectory.last_frame - trajectory.first_frame
            ) / max(self.config.fps, 1)

            group = next(
                (
                    group
                    for group in groups
                    if visitor_id in group.members
                ),
                None,
            )

            dwell = (
                self.zone_engine.get_dwell(
                    visitor_id,
                    self.config.fps,
                )
                if self.config.enable_zones
                else {}
            )

            people.append({
                "visitor_id": visitor_id,
                "role": roles.get(visitor_id, "Customer"),
                "group_id": group.group_id if group else None,
                "group_status": group.status if group else None,
                "first_seen_frame": trajectory.first_frame,
                "last_seen_frame": trajectory.last_frame,
                "duration_seconds": round(duration, 2),
                "total_observations": len(trajectory.positions),
                "zones_dwell_seconds": dwell,
            })

        return {
            "pipeline_version": "13.2.0",
            "analysis_timestamp": datetime.now().isoformat(),
            "total_frames_processed": self.frame_idx,
            "duration_seconds": round(
                self.frame_idx / max(self.config.fps, 1),
                2,
            ),
            "group_settings": {
                "distance_threshold_px": self.config.group_distance_threshold_px,
                "minimum_together_seconds": self.config.group_min_together_seconds,
                "proximity_ratio": self.config.group_proximity_ratio,
                "likely_group_video_fraction": self.config.group_likely_video_fraction,
                "continuous_required": self.config.group_require_continuous,
                "max_gap_frames": self.config.group_max_gap_frames,
            },
            "summary": {
                "total_distinct_people": len(people),
                "customers_detected": customer_count,
                "staff_cashiers_detected": staff_count,
                "groups_detected": len(groups),
                "likely_groups_detected": sum(
                    1
                    for group in groups
                    if group.status == "likely_group"
                ),
                "peak_occupancy": self.peak_occupancy,
                "total_entries": self.entrance_engine.entry_count,
                "total_exits": self.entrance_engine.exit_count,
            },
            "groups": [
                {
                    "group_id": group.group_id,
                    "status": group.status,
                    "members": [
                        f"V{member}"
                        for member in group.members
                    ],
                    "member_count": group.size,
                    "evidence": group.evidence,
                }
                for group in groups
            ],
            "all_pair_evidence": [
                {
                    "pair": [
                        f"V{pair[0]}",
                        f"V{pair[1]}",
                    ],
                    "evidence": evidence,
                }
                for pair, evidence in pair_evidence.items()
            ],
            "people": people,
            "entrance_events": self.entrance_engine.events[:100],
            "occupancy_timeline": self.occupancy_timeline,
        }

    def render_final_group_video(
        self,
        source: str,
    ):
        if not self.config.output_video:
            return

        if not self.frame_records:
            print("[!] No frame records available for final video rendering.")
            return

        print(
            f"[-] Rendering final annotated video with group labels: "
            f"{self.config.output_video_path}"
        )

        cap = cv2.VideoCapture(
            int(source)
            if source.isdigit()
            else source
        )

        if not cap.isOpened():
            print("[!] Cannot reopen source video for final rendering.")
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            self.config.output_video_path,
            fourcc,
            self.config.fps,
            (width, height),
        )

        group_by_visitor = {}

        for group in self.final_groups:
            for visitor_id in group.members:
                group_by_visitor[visitor_id] = group

        frame_number = 0

        try:
            while cap.isOpened() and frame_number < len(self.frame_records):
                ret, frame = cap.read()

                if not ret:
                    break

                frame_record = self.frame_records[frame_number]

                annotated = self._draw_frame(
                    frame=frame,
                    frame_record=frame_record,
                    frame_idx=frame_number,
                    current_occupancy=len(frame_record),
                    group_by_visitor=group_by_visitor,
                    final_render=True,
                )

                writer.write(annotated)

                if frame_number > 0 and frame_number % 100 == 0:
                    print(
                        f"  Final video rendering: "
                        f"{frame_number}/{len(self.frame_records)} frames"
                    )

                frame_number += 1

        finally:
            cap.release()
            writer.release()

        print(
            f"[+] Annotated Video: {self.config.output_video_path}"
        )

    def save_debug_log(self):
        if (
            not self.config.debug_identity_log
            or not self.debug_log
        ):
            return

        import csv

        keys = list(self.debug_log[0].keys())

        with open(
            self.config.debug_log_path,
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=keys,
            )

            writer.writeheader()
            writer.writerows(self.debug_log)

        print(
            f"[+] Identity debug log: "
            f"{self.config.debug_log_path}"
        )

        self.gallery.dump_similarity_stats(
            self.config.similarity_stats_path
        )

        print(
            f"[+] Similarity stats: "
            f"{self.config.similarity_stats_path}"
        )

    def run(self, source: str = "0"):
        cap = cv2.VideoCapture(
            int(source)
            if source.isdigit()
            else source
        )

        if not cap.isOpened():
            print(f"[!] Cannot open video source: {source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0:
            self.config.fps = fps

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(
            f"[+] Video: {width}x{height} @ "
            f"{self.config.fps:.1f} FPS "
            f"({total_frames} frames)"
        )

        print(
            f"[+] Gallery: {self.config.gallery_rolling_size} embeddings/person "
            f"@ {self.config.gallery_sample_rate_hz}Hz | "
            f"Match threshold: {self.config.gallery_match_threshold}"
        )

        together_mode = (
            "continuous"
            if self.config.group_require_continuous
            else "accumulated"
        )

        print(
            f"[+] Groups: distance<{self.config.group_distance_threshold_px}px | "
            f"min {self.config.group_min_together_seconds:.1f}s "
            f"({together_mode}) | "
            f"ratio>={self.config.group_proximity_ratio:.2f} | "
            f"likely>={self.config.group_likely_video_fraction:.2f} "
            f"of video\n"
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                annotated = self.process_frame(frame)

                if self.config.show_preview:
                    cv2.imshow(
                        "Retail CCTV Analytics v13.2",
                        annotated,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[!] Stopped manually by user.")
                        break

                if self.frame_idx % 100 == 0:
                    elapsed = time.time() - self.start_time

                    process_fps = (
                        self.frame_idx
                        / max(elapsed, 0.001)
                    )

                    percent = (
                        self.frame_idx / total_frames * 100
                        if total_frames > 0
                        else 0
                    )

                    print(
                        f"  Frame {self.frame_idx} "
                        f"({percent:.1f}%) | "
                        f"{process_fps:.1f} FPS | "
                        f"Visitors: {self.gallery.distinct_count()} | "
                        f"Candidates: {len(self.gallery.candidates)}"
                    )

        finally:
            cap.release()
            cv2.destroyAllWindows()

            if os.path.exists(self.tracker_config_path):
                os.remove(self.tracker_config_path)

            print("\n" + "=" * 62)
            print("  RETAIL ANALYTICS REPORT")
            print("=" * 62)

            report = self.generate_report()

            def clean(value):
                if isinstance(value, dict):
                    return {
                        key: clean(val)
                        for key, val in value.items()
                    }

                if isinstance(value, list):
                    return [clean(item) for item in value]

                if isinstance(value, tuple):
                    return list(value)

                if isinstance(value, set):
                    return list(value)

                if isinstance(value, np.ndarray):
                    return value.tolist()

                if isinstance(value, np.integer):
                    return int(value)

                if isinstance(value, np.floating):
                    return float(value)

                return value

            if self.config.output_json:
                with open(
                    self.config.output_json_path,
                    "w",
                ) as f:
                    json.dump(
                        clean(report),
                        f,
                        indent=2,
                        default=str,
                    )

                print(
                    f"[+] JSON Report: "
                    f"{self.config.output_json_path}"
                )

            self.save_debug_log()

            summary = report["summary"]

            print(f"\n  Total Frames:        {report['total_frames_processed']}")
            print(f"  Duration:            {report['duration_seconds']}s")
            print(f"  Total People:        {summary['total_distinct_people']}")
            print(f"    - Customers:       {summary['customers_detected']}")
            print(f"    - Staff/Cashiers:  {summary['staff_cashiers_detected']}")
            print(f"  Groups Detected:     {summary['groups_detected']}")
            print(f"  Likely Groups:       {summary['likely_groups_detected']}")

            for group in report["groups"]:
                evidence = group["evidence"]

                print(
                    f"    * Group {group['group_id']} "
                    f"[{group['status']}]: "
                    f"{group['members']} | "
                    f"together: {evidence.get('total_close_seconds', '?')}s | "
                    f"continuous: {evidence.get('longest_continuous_close_seconds', '?')}s | "
                    f"video ratio: {evidence.get('video_together_ratio', '?')}"
                )

            print(f"  Peak Occupancy:      {summary['peak_occupancy']}")

            if self.config.enable_entrance_line:
                print(
                    f"  Entries / Exits:     "
                    f"{summary['total_entries']} / "
                    f"{summary['total_exits']}"
                )

            print("=" * 62 + "\n")

            # Final video is rendered after final group detection.
            # Therefore G1/G2 labels are accurate.
            self.render_final_group_video(source)


# =====================================================================
# ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        type=str,
        default="video.mp4",
    )

    parser.add_argument(
        "--detector",
        type=str,
        default="yolo11x.pt",
        help="Example: yolo11s.pt | yolo11m.pt | yolo11l.pt | yolo11x.pt",
    )

    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=0,
        help="0 uses Ultralytics default. Try 960 for improved small-person detection.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Gallery ReID match similarity threshold",
    )

    parser.add_argument(
        "--rolling-size",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--group-dist",
        type=float,
        default=140.0,
        help="Maximum pixel distance between group members",
    )

    parser.add_argument(
        "--group-ratio",
        type=float,
        default=0.60,
        help="Minimum close ratio over shared visible frames",
    )

    parser.add_argument(
        "--group-min-seconds",
        type=float,
        default=5.0,
        help="Minimum together duration before creating a group",
    )

    parser.add_argument(
        "--group-likely-fraction",
        type=float,
        default=0.30,
        help="Together fraction of full video needed for likely_group",
    )

    parser.add_argument(
        "--group-gap-frames",
        type=int,
        default=3,
        help="Allowed missing-detection gap inside a continuous group period",
    )

    parser.add_argument(
        "--group-total-time",
        action="store_true",
        help="Use accumulated together time instead of continuous together time",
    )

    parser.add_argument(
        "--group-include-staff",
        action="store_true",
        help="Allow staff/cashiers to be included in groups",
    )

    parser.add_argument(
        "--min-track-frames",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--enable-entrance",
        action="store_true",
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live preview window during first processing pass",
    )

    parser.add_argument(
        "--show-track-id",
        action="store_true",
        help="Overlay BoT-SORT track ID along with visitor ID",
    )

    parser.add_argument(
        "--output-video",
        type=str,
        default="annotated_output.mp4",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="analytics_output.json",
    )

    parser.add_argument(
        "--debug-log",
        type=str,
        default="identity_debug.csv",
    )

    args = parser.parse_args()

    config = PipelineConfig(
        detector_model=args.detector,
        detector_confidence=args.conf,
        detector_imgsz=args.imgsz,
        tracker_type=args.tracker,
        gallery_match_threshold=args.threshold,
        gallery_rolling_size=args.rolling_size,
        group_distance_threshold_px=args.group_dist,
        group_proximity_ratio=args.group_ratio,
        group_min_together_seconds=args.group_min_seconds,
        group_likely_video_fraction=args.group_likely_fraction,
        group_max_gap_frames=args.group_gap_frames,
        group_require_continuous=not args.group_total_time,
        group_include_staff=args.group_include_staff,
        min_track_frames=args.min_track_frames,
        enable_entrance_line=args.enable_entrance,
        show_preview=args.preview,
        show_tracker_id=args.show_track_id,
        output_video_path=args.output_video,
        output_json_path=args.output_json,
        debug_log_path=args.debug_log,
    )

    pipeline = RetailAnalyticsPipeline(config=config)
    pipeline.run(source=args.source)
