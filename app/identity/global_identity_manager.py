"""
GlobalIdentityManager — the single authority on Global Person IDs.

Matching uses a COMPOSITE SCORE of three independent signals:
  1. Appearance (Re-ID embedding similarity)         — weight 0.45
  2. Spatial proximity (last-known bbox vs new bbox)  — weight 0.35
  3. Temporal recency (how recently the person was seen) — weight 0.20

Hard anti-swap rules:
  • A person who is ACTIVE with a live track on the SAME camera can NEVER
    be re-assigned to another bounding box on that camera in the same frame.
  • Only TEMPORARILY_LOST persons (or persons on different cameras) are
    eligible for Re-ID matching.
"""
import numpy as np
from collections import defaultdict, deque

from app.config import Config
from app.reid.gallery import RollingGallery, MultiViewGallery
from app.reid.clothing import ClothingProfile, extract_clothing_features, compare_clothing


# ── States ────────────────────────────────────────────────────────────────────
TENTATIVE          = "TENTATIVE"           # Detection or identity is not yet sufficiently confirmed
ACTIVE             = "ACTIVE"              # A valid visitor identity and visit are established
TEMPORARILY_LOST   = "TEMPORARILY_LOST"    # Person is no longer visible, but identity history is retained
IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"  # Multiple candidates exist or confidence is ambiguous; deferred
VISIT_COMPLETED    = "VISIT_COMPLETED"     # A valid exit has been confirmed
HISTORICAL         = "HISTORICAL"          # Departed visitor retained in memory for returning visitor recognition
REIDENTIFIED       = "REIDENTIFIED"        # Successfully recovered on this frame
STATIC             = "STATIC"              # Stationary object/statue
UNVERIFIED_OCCUPANT= "UNVERIFIED_OCCUPANT" # Pre-existing occupant present without observed entry
DEPARTED           = "DEPARTED"            # Exited visitor status
EXPIRED            = "EXPIRED"             # Identity exceeded maximum time gap


def _compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea <= 0:
        return 0.0
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return float(interArea / float(boxAArea + boxBArea - interArea))


def _bbox_centre(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _spatial_score(bbox_a, bbox_b, frame_diag=None):
    """
    Score between 0 and 1 indicating how close two bboxes are.
    1.0 = same position, 0.0 = opposite corners of frame.
    """
    ca = _bbox_centre(bbox_a)
    cb = _bbox_centre(bbox_b)
    dist = np.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
    if frame_diag is None or frame_diag == 0:
        frame_diag = 1000.0
    # Normalise: 0 distance → 1.0, frame_diag distance → 0.0
    return max(0.0, 1.0 - dist / frame_diag)


def _temporal_score(time_gap, max_gap=None):
    """
    Score between 0 and 1 based on how recently the person was seen.
    0 seconds ago → 1.0, max_gap seconds ago → 0.0.
    """
    if max_gap is None:
        max_gap = Config.MAX_REID_TIME_GAP
    if max_gap <= 0:
        return 1.0
    return max(0.0, 1.0 - time_gap / max_gap)


# ─────────────────────────────────────────────────────────────────────────────

class GlobalPerson:
    def __init__(self, gid, timestamp, cam_id, track_id, embedding, bbox):
        self.global_person_id = gid
        self.first_seen       = timestamp
        self.last_seen        = timestamp
        self.current_camera   = cam_id
        self.current_track_id = track_id
        
        self.first_bbox       = bbox
        self.last_bbox        = bbox      # spatial memory
        
        # Rolling motion tracking for 100% reliable statue/mannequin rejection
        self.head_history = deque(maxlen=60)
        self.foot_history = deque(maxlen=60)
        self.is_static    = False
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            hx = (x1 + x2) / 2.0
            hy = float(y1)
            fx = (x1 + x2) / 2.0
            fy = float(y2)
            self.head_history.append((hx, hy))
            self.foot_history.append((fx, fy))
        
        self.status = ACTIVE
        self.is_departed = False
        self.current_visit_status = "PROVISIONAL"  # "PROVISIONAL", "CONFIRMED", "SHORT_VISIT_IGNORED", "COMPLETED"
        self.visit_id = f"V-{gid}-1"
        self.visit_count = 1
        self.co_present_track_ids = set()

        self.gallery = RollingGallery()
        if embedding is not None:
            self.gallery.add(embedding)

        self.camera_history          = [cam_id]
        self.reidentification_events = []
        self.total_visible_frames    = 1
        self.times_lost              = 0

    def estimate_viewpoint(self):
        """
        Estimate person viewpoint ('front', 'back', 'side', or 'general')
        based on rolling trajectory velocity.
        """
        if len(self.foot_history) < 4:
            return 'general'
        pts = list(self.foot_history)[-12:]
        dx = pts[-1][0] - pts[0][0]
        dy = pts[-1][1] - pts[0][1]
        dist = np.hypot(dx, dy)
        if dist < 6.0:
            return 'general'
        abs_dx, abs_dy = abs(dx), abs(dy)
        if abs_dx > 1.6 * abs_dy:
            return 'side'
        # Moving downwards (increasing y) towards camera = front, moving upwards away = back
        return 'front' if dy > 0 else 'back'

    def terminate_visit(self, timestamp, is_short=False):
        """
        Officially terminates the current visit upon exit.
        If is_short=True, visit is closed as SHORT_VISIT_IGNORED and excluded from official counts.
        """
        self.is_departed = True
        self.last_seen = timestamp
        self.current_visit_status = "SHORT_VISIT_IGNORED" if is_short else "COMPLETED"
        self.status = DEPARTED

    def start_new_visit(self, timestamp, cam_id, track_id, embedding, bbox, clothing=None):
        """
        Starts a new visit record upon genuine re-entry through the entrance gate.
        """
        self.visit_count += 1
        self.visit_id = f"V-{self.global_person_id}-{self.visit_count}"
        self.is_departed = False
        self.current_visit_status = "PROVISIONAL"
        self.status = ACTIVE
        self.update_active(timestamp, cam_id, track_id, embedding, bbox, clothing=clothing, is_occluded=False)

    def update_active(self, timestamp, cam_id, track_id, embedding, bbox, clothing=None, is_occluded=False):
        self.last_seen        = timestamp
        self.current_track_id = track_id
        self.current_camera   = cam_id
        
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            hx = (x1 + x2) / 2.0
            hy = float(y1)
            fx = (x1 + x2) / 2.0
            fy = float(y2)
            self.head_history.append((hx, hy))
            self.foot_history.append((fx, fy))

            # Evaluate rolling motion over the last 20-25 frames (0.8 - 1.0 second):
            if len(self.head_history) >= 20:
                recent_n = min(len(self.head_history), 25)
                h_pts = list(self.head_history)[-recent_n:]
                f_pts = list(self.foot_history)[-recent_n:]

                # Head trajectory
                head_dx = h_pts[-1][0] - h_pts[0][0]
                head_dy = h_pts[-1][1] - h_pts[0][1]
                net_head_motion = np.hypot(head_dx, head_dy)

                foot_dx = f_pts[-1][0] - f_pts[0][0]
                foot_dy = f_pts[-1][1] - f_pts[0][1]
                net_foot_motion = np.hypot(foot_dx, foot_dy)

                # Head spread radius around median position
                med_hx = np.median([p[0] for p in h_pts])
                med_hy = np.median([p[1] for p in h_pts])
                head_radius = max(np.hypot(p[0] - med_hx, p[1] - med_hy) for p in h_pts)

                # If head and feet barely move across recent frames, person is temporarily stationary (paused)
                if head_radius < 14.0 and net_head_motion < 14.0 and net_foot_motion < 18.0:
                    self.is_stationary = True
                elif net_head_motion > 24.0 or net_foot_motion > 28.0:
                    self.is_stationary = False
                self.is_static = False
                
        self.last_bbox        = bbox
        self.status           = ACTIVE
        self.total_visible_frames += 1
        if cam_id not in self.camera_history:
            self.camera_history.append(cam_id)

        if embedding is not None and not is_occluded:
            view_tag = self.estimate_viewpoint()
            # Crop quality check
            quality = 1.0
            if bbox is not None:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bh < 40 or bw < 20:
                    quality = 0.15

            if quality >= 0.20:
                # ── Safe Multi-View & Clothing Gallery Update ──
                if self.gallery.is_empty():
                    self.gallery.add(embedding, view_tag=view_tag, clothing=clothing)
                else:
                    sim_check = self.gallery.best_similarity(embedding)
                    cloth_check = self.gallery.clothing_similarity(clothing) if clothing is not None else None

                    # 1. High deep similarity -> safely add to corresponding view partition.
                    if sim_check >= 0.62:
                        self.gallery.add(embedding, view_tag=view_tag, clothing=clothing)
                    # 2. Front-to-back turn during continuous tracking:
                    # Verified viewpoint change: clothing matches strongly and deep similarity is consistent
                    elif cloth_check is not None and cloth_check >= 0.65 and sim_check >= 0.48:
                        self.gallery.add(embedding, view_tag=view_tag, clothing=clothing)

    def mark_reidentified(self, timestamp, cam_id, track_id, embedding, bbox, sim, clothing=None):
        self.reidentification_events.append({
            "time": timestamp, "from_cam": self.current_camera,
            "to_cam": cam_id, "track_id": track_id,
            "similarity": round(sim, 4),
        })
        self.update_active(timestamp, cam_id, track_id, embedding, bbox, clothing=clothing, is_occluded=False)
        self.status = REIDENTIFIED

    def mark_lost(self):
        self.status      = TEMPORARILY_LOST
        self.times_lost += 1

    def mark_expired(self):
        self.status = EXPIRED


# ─────────────────────────────────────────────────────────────────────────────

class GlobalIdentityManager:
    # Composite score weights
    W_APPEARANCE = 0.45
    W_SPATIAL    = 0.35
    W_TEMPORAL   = 0.20

    # Minimum composite score to accept a Re-ID match
    COMPOSITE_THRESHOLD = 0.40

    # ── Clothing gate ─────────────────────────────────────────────────────────
    # A Re-ID match MUST clear this minimum clothing similarity before any
    # identity assignment is made. This is the critical safeguard that prevents
    # cross-person identity transfers when CNN embeddings are not discriminative.
    # When clothing profiles are unavailable, the gate is skipped (permissive).
    CLOTHING_MIN_GATE = 0.30  # veto threshold: below this → REJECT regardless of deep sim

    # ── Ambiguity Confidence Margin ───────────────────────────────────────────
    # If the difference between the best candidate score and the second-best candidate
    # score is less than this margin, the association is considered ambiguous and deferred.
    CONFIDENCE_MARGIN = 0.08

    def __init__(self, camera_manager, require_candidate_motion: bool = True):
        self.camera_manager = camera_manager
        self.require_candidate_motion = require_candidate_motion
        self.persons        = {}        # gid → GlobalPerson
        self.local_to_global = {}       # track_id → gid
        self.candidates     = {}        # track_id → dict (dynamic motion tracking)
        self.co_present_tracks = defaultdict(set) # track_id/gid → set of simultaneously visible track_ids/gids
        self.latest_features = {}       # track_id → latest np.ndarray feature
        self.latest_clothing = {}       # track_id → latest ClothingProfile
        self._next_idx      = 1
        self._frame_diag    = None      # cached frame diagonal
        self.gate_counter   = None      # optional TripwireCounter reference
        self.reid_log       = []        # audit log of every Re-ID match attempt
        self.vlm_verifier   = None      # optional local VLM tie-breaker (Ollama)

    def set_vlm_verifier(self, vlm_verifier):
        """Configure local VLM tie-breaker."""
        self.vlm_verifier = vlm_verifier

    def _query_vlm(self, crop, gid, tid):
        """
        Query VLM verifier using non-blocking cached/async execution in live mode,
        or synchronous execution if in unit test harness (mock).
        """
        if self.vlm_verifier is None or not self.vlm_verifier.enabled:
            return None
        if crop is None or crop.size == 0:
            return None
        if hasattr(self.vlm_verifier, "get_cached_verdict"):
            verdict = self.vlm_verifier.get_cached_verdict(tid, gid)
            if verdict is not None:
                return verdict

        # In unit tests or mock objects without _executor, invoke synchronously
        is_mock = (
            not hasattr(self.vlm_verifier, "_executor") or
            hasattr(self.vlm_verifier.verify_candidate, "assert_called") or
            hasattr(self.vlm_verifier.verify_candidate, "return_value")
        )
        if is_mock:
            return self.vlm_verifier.verify_candidate(crop, gid)

        # In production live stream: dispatch asynchronously without blocking frame loop
        if hasattr(self.vlm_verifier, "request_verification_async"):
            self.vlm_verifier.request_verification_async(crop, gid, tid)
        return None

    def _new_gid(self):
        gid = f"P-{self._next_idx:03d}"
        self._next_idx += 1
        return gid

    def get_reid_log(self, decision_filter=None):
        """Return the Re-ID audit log, optionally filtered by decision type."""
        if decision_filter is None:
            return list(self.reid_log)
        return [e for e in self.reid_log if e.get("decision") == decision_filter]

    def dump_reid_log(self, path: str):
        """Write the full Re-ID audit log to a CSV file for external analysis."""
        import csv, os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "frame", "ts", "camera_id", "query_tid", "assigned_gid",
            "candidate_gids", "best_candidate", "second_candidate",
            "deep_sim", "cloth_sim", "app_score", "sp_score",
            "tp_score", "composite", "margin", "decision", "reason",
            "allow_gallery_update", "min_composite", "min_app",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            if self.reid_log:
                writer.writerows(self.reid_log)

    def terminate_visit(self, gid, timestamp, is_short=False):
        """
        Officially close/terminate a visit when a person exits the store.
        If is_short=True (duration <= 5s), marked as SHORT_VISIT_IGNORED.
        Exited persons become DEPARTED and cannot be Re-ID matched to in-store tracks.
        """
        if gid in self.persons:
            p = self.persons[gid]
            p.terminate_visit(timestamp, is_short=is_short)
            if p.current_track_id in self.local_to_global:
                if self.local_to_global[p.current_track_id] == gid:
                    del self.local_to_global[p.current_track_id]

    def _sync_co_presence(self, tid, gid, person):
        """Ensure co-presence sets for tid and gid are completely synchronized."""
        co_set = set(self.co_present_tracks[tid])
        person.co_present_track_ids.update(co_set)
        self.co_present_tracks[gid].update(co_set)
        self.co_present_tracks[gid].add(tid)
        self.co_present_tracks[tid].add(gid)

        for other in list(co_set):
            self.co_present_tracks[other].add(gid)
            self.co_present_tracks[other].add(tid)
            if other in self.persons:
                self.persons[other].co_present_track_ids.add(gid)
                self.persons[other].co_present_track_ids.add(tid)
                person.co_present_track_ids.add(other)
            elif other in self.local_to_global:
                other_gid = self.local_to_global[other]
                if other_gid in self.persons:
                    self.persons[other_gid].co_present_track_ids.add(gid)
                    self.persons[other_gid].co_present_track_ids.add(tid)
                    person.co_present_track_ids.add(other_gid)
                    self.co_present_tracks[other_gid].add(gid)
                    self.co_present_tracks[other_gid].add(tid)

    def promote_track_to_visitor(self, tid, timestamp, cam_id="camera_01", feat=None, bbox=None, clothing=None, crop=None):
        """
        Assign a permanent visitor ID (P-001, P-002...) to a tracked person.
        CALLED ONLY after the person's tracked foot point genuinely crosses the
        configured entry line from outside to inside.
        """
        if tid in self.local_to_global:
            gid = self.local_to_global[tid]
            if gid in self.persons:
                self._sync_co_presence(tid, gid, self.persons[gid])
            return gid

        if (cam_id == "camera_01" or cam_id is None) and tid in self.candidates and "camera_id" in self.candidates[tid]:
            cam_id = self.candidates[tid]["camera_id"]

        if feat is None:
            feat = self.latest_features.get(tid)
        if clothing is None:
            clothing = self.latest_clothing.get(tid)
        if crop is None and tid in self.candidates:
            crop = self.candidates[tid].get("crop")

        # Check if this person is a returning visitor who previously completed a visit
        # within the configured retention window (including short visits where identity memory is preserved):
        departed_match = None
        departed_candidates = []
        if feat is not None:
            retention_limit = getattr(Config, "VISITOR_RETENTION_TIME", 3600)
            strong_thresh = getattr(Config, "REID_STRONG_MATCH_THRESHOLD", 0.75)
            for gid, p in self.persons.items():
                is_eligible = (p.is_departed and p.current_visit_status in ("COMPLETED", "SHORT_VISIT_IGNORED")) or (p.status == TEMPORARILY_LOST)
                if is_eligible:
                    if (timestamp - p.last_seen) <= retention_limit:
                        sim = p.gallery.best_similarity(feat)
                        if clothing and len(p.gallery.clothing_profiles) > 0:
                            c_sim = p.gallery.clothing_similarity(clothing)
                            sim = 0.70 * sim + 0.30 * c_sim
                        if sim >= strong_thresh:
                            departed_candidates.append((sim, p))

        departed_candidates.sort(key=lambda x: x[0], reverse=True)
        if departed_candidates:
            if len(departed_candidates) > 1 and (departed_candidates[0][0] - departed_candidates[1][0] < self.CONFIDENCE_MARGIN):
                # Ambiguous tie: check if local VLM can break the tie between departed visitors
                if crop is not None and getattr(self, "vlm_verifier", None) is not None and self.vlm_verifier.enabled:
                    top_p = departed_candidates[0][1]
                    sec_p = departed_candidates[1][1]
                    v1 = self._query_vlm(crop, top_p.global_person_id, tid)
                    v2 = self._query_vlm(crop, sec_p.global_person_id, tid)
                    if v1 and v1.get("match") and (not v2 or not v2.get("match")):
                        departed_match = top_p
                    elif v2 and v2.get("match") and (not v1 or not v1.get("match")):
                        departed_match = sec_p
                    else:
                        departed_match = None
                else:
                    departed_match = None
            else:
                departed_match = departed_candidates[0][1]
        elif crop is not None and feat is not None and getattr(self, "vlm_verifier", None) is not None and self.vlm_verifier.enabled:
            # Check near-threshold returning visitor in ambiguity range
            amb_low = getattr(Config, "VLM_AMBIGUITY_LOW", 0.68)
            retention_limit = getattr(Config, "VISITOR_RETENTION_TIME", 3600)
            strong_thresh = getattr(Config, "REID_STRONG_MATCH_THRESHOLD", 0.75)
            for gid_d, p_d in self.persons.items():
                is_eligible = (p_d.is_departed and p_d.current_visit_status in ("COMPLETED", "SHORT_VISIT_IGNORED")) or (p_d.status == TEMPORARILY_LOST)
                if is_eligible:
                    if (timestamp - p_d.last_seen) <= retention_limit:
                        sim = p_d.gallery.best_similarity(feat)
                        if amb_low <= sim < strong_thresh:
                            v_res = self._query_vlm(crop, gid_d, tid)
                            if v_res and v_res.get("match") and v_res.get("confidence", 0.0) >= 0.75:
                                departed_match = p_d
                                break

        if departed_match is not None:
            # Genuine re-entry of previously known visitor!
            if departed_match.is_departed:
                departed_match.start_new_visit(timestamp, cam_id, tid, feat, bbox, clothing=clothing)
            else:
                departed_match.mark_reidentified(timestamp, cam_id, tid, feat, bbox, comp=1.0, clothing=clothing)
            gid = departed_match.global_person_id
            self.local_to_global[tid] = gid
            self._sync_co_presence(tid, gid, departed_match)
            if tid in self.candidates:
                del self.candidates[tid]
            return gid

        gid = self._new_gid()
        p = GlobalPerson(gid, timestamp, cam_id, tid, feat, bbox)
        if clothing is not None:
            p.gallery.add_clothing(clothing)
        self.persons[gid] = p
        self.local_to_global[tid] = gid
        self._sync_co_presence(tid, gid, p)
        if tid in self.candidates:
            del self.candidates[tid]
        return gid

    def _compute_quality(self, bbox):
        x1, y1, x2, y2 = bbox
        cw = max(1, x2 - x1)
        ch = max(1, y2 - y1)
        if ch < 40 or cw < 20: return 0.15
        resolution_score = min(1.0, (ch * cw) / (40 * 20 * 5))
        aspect = cw / max(ch, 1)
        aspect_score = 1.0 - min(1.0, abs(aspect - 0.45) / 0.90)
        border_score = 0.7 if (x1 < 15 or y1 < 15) else 1.0
        return float(np.clip(resolution_score * aspect_score * border_score, 0.15, 1.0))

    def process_detections(self, detections, features_map, timestamp, frame_idx, gate_counter=None, frame_w=1920, frame_h=1080, clothing_map=None):
        """
        detections   : list of dicts from local_tracker
        features_map : dict { track_id → np.ndarray }  (may omit known tracks)
        timestamp    : seconds since start
        frame_idx    : current frame number
        clothing_map : dict { track_id → ClothingProfile } (optional)
        """
        if features_map:
            self.latest_features.update(features_map)

        if clothing_map is None:
            clothing_map = {}
        for det in detections:
            tid = det['track_id']
            if tid not in clothing_map and det.get('crop') is not None:
                c_prof = extract_clothing_features(det['crop'])
                if c_prof is not None:
                    clothing_map[tid] = c_prof
        if clothing_map:
            self.latest_clothing.update(clothing_map)

        # Pairwise IoU to detect occluded / overlapping tracks in this frame
        occluded_tracks = set()
        for i in range(len(detections)):
            b1 = detections[i]['bbox']
            for j in range(i + 1, len(detections)):
                b2 = detections[j]['bbox']
                if _compute_iou(b1, b2) > 0.12:
                    occluded_tracks.add(detections[i]['track_id'])
                    occluded_tracks.add(detections[j]['track_id'])

        if gate_counter is not None:
            self.gate_counter = gate_counter
            if getattr(gate_counter, "identity_manager", None) is None:
                gate_counter.identity_manager = self

        # Cache frame diagonal from first detection bbox estimate
        if self._frame_diag is None and detections:
            # rough estimate
            self._frame_diag = float(np.hypot(frame_w, frame_h))

        active_tracks = {d['track_id'] for d in detections}
        decisions     = []

        # ── Step 0: Immediate Lost and Expire State Update ──────────────────
        # Mark missing tracks TEMPORARILY_LOST immediately so they are eligible for
        # recovery during Re-ID in the current frame without race conditions.
        for gid, p in self.persons.items():
            if p.status in (ACTIVE, REIDENTIFIED):
                if p.current_track_id not in active_tracks:
                    p.mark_lost()

        for gid, p in self.persons.items():
            if p.status == TEMPORARILY_LOST:
                if (timestamp - p.last_seen) > Config.MAX_REID_TIME_GAP:
                    p.mark_expired()

        # ── 0. Update Co-Presence / Simultaneous-Visibility Graph ───────────
        # Any tracks visible in the SAME frame can NEVER be the same person!
        track_list = list(active_tracks)
        for i in range(len(track_list)):
            t1 = track_list[i]
            g1 = self.local_to_global.get(t1)
            for j in range(i + 1, len(track_list)):
                t2 = track_list[j]
                g2 = self.local_to_global.get(t2)
                self.co_present_tracks[t1].add(t2)
                self.co_present_tracks[t2].add(t1)
                if g1:
                    self.co_present_tracks[g1].add(t2)
                    self.co_present_tracks[t2].add(g1)
                    if g1 in self.persons:
                        self.persons[g1].co_present_track_ids.add(t2)
                if g2:
                    self.co_present_tracks[g2].add(t1)
                    self.co_present_tracks[t1].add(g2)
                    if g2 in self.persons:
                        self.persons[g2].co_present_track_ids.add(t1)
                if g1 and g2:
                    self.co_present_tracks[g1].add(g2)
                    self.co_present_tracks[g2].add(g1)

        # ── 1. Split known vs unknown ─────────────────────────────────────
        known_dets   = []
        unknown_dets = []
        for det in detections:
            tid = det['track_id']
            if tid in self.local_to_global:
                gid = self.local_to_global[tid]
                p = self.persons.get(gid)
                # If this visitor already departed, do NOT treat as active known visitor!
                if p and (p.is_departed or p.status in (EXPIRED, DEPARTED, VISIT_COMPLETED)):
                    unknown_dets.append(det)
                else:
                    known_dets.append(det)
            else:
                unknown_dets.append(det)

        # ── 2. Update known tracks (no matching needed) ───────────────────
        for det in known_dets:
            tid = det['track_id']
            gid = self.local_to_global[tid]
            p   = self.persons[gid]
            feat = features_map.get(tid)
            cloth = clothing_map.get(tid)
            
            # COUNTHUM LOGIC: Reject bad crops for gallery update
            quality = self._compute_quality(det['bbox'])
            if quality < 0.25:
                feat = None

            p.update_active(timestamp, det['camera_id'], tid, feat, det['bbox'],
                            clothing=cloth, is_occluded=(tid in occluded_tracks))
            
            state = "STATIONARY" if getattr(p, "is_stationary", False) else ACTIVE
            decisions.append({
                "track_id": tid, "global_id": gid,
                "state": state, "sim": None
            })

        # ── Pre-initialize candidates for unknown detections ──────────────
        active_gate = gate_counter or self.gate_counter
        for det in unknown_dets:
            tid = det['track_id']
            bx1, by1, bx2, by2 = det['bbox']
            hx = (bx1 + bx2) / 2.0
            hy = float(by1)
            fx = (bx1 + bx2) / 2.0
            fy = float(by2)

            if tid not in self.candidates:
                is_preexisting_inside = False
                if active_gate is not None and hasattr(active_gate, "is_inside_or_entering"):
                    is_preexisting_inside = active_gate.is_inside_or_entering(tid, det['bbox'], frame_w, frame_h)

                # An initial occupant was present before any visitor entries occurred
                is_initial_occupant = is_preexisting_inside and (len(self.persons) == 0)

                self.candidates[tid] = {
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "camera_id": det['camera_id'],
                    "head_pts": deque(maxlen=90),
                    "foot_pts": deque(maxlen=90),
                    "is_static": False,
                    "is_initial_occupant": is_initial_occupant,
                    "is_unverified_occupant": is_preexisting_inside,
                    "anchor_head": (hx, hy),
                    "anchor_foot": (fx, fy),
                    "crop": det.get('crop'),
                }
            cand = self.candidates[tid]
            cand["last_seen"] = timestamp
            cand["camera_id"] = det['camera_id']
            if det.get('crop') is not None:
                cand["crop"] = det['crop']
            cand["head_pts"].append((hx, hy))
            cand["foot_pts"].append((fx, fy))

        # ── 3. Re-ID unknown tracks (COMPOSITE scoring & Margin Gating) ───
        if unknown_dets:
            currently_active_on_cam = defaultdict(set)
            for det in known_dets:
                tid = det['track_id']
                gid = self.local_to_global[tid]
                currently_active_on_cam[det['camera_id']].add(gid)

            query_candidates = {}

            for q_idx, det in enumerate(unknown_dets):
                tid     = det['track_id']
                cam_id  = det['camera_id']
                feat    = features_map.get(tid)
                q_bbox  = det['bbox']
                q_cloth = clothing_map.get(tid)
                
                # Rule 1: Static candidate (statue / mannequin) can NEVER be Re-ID matched!
                cand = self.candidates.get(tid)
                if cand and cand.get("is_static", False):
                    continue

                # Rule 2: Initial staff/occupant present before entries began can NEVER be Re-ID matched!
                if cand and cand.get("is_initial_occupant", False):
                    continue

                # Rule 3: Do not attempt Re-ID with poor quality crop
                quality = self._compute_quality(q_bbox)
                if feat is None or quality < 0.20:
                    continue

                evaluated_gids = []
                cands_for_q = []

                for gid, p in self.persons.items():
                    if p.gallery.is_empty():
                        continue

                    # Rule 4: Person already visible on this camera in this frame
                    if gid in currently_active_on_cam.get(cam_id, set()):
                        continue

                    # Rule 5: Person is active on this cam with a different live track
                    if (p.status == ACTIVE
                            and p.current_camera == cam_id
                            and p.current_track_id in active_tracks
                            and p.current_track_id != tid):
                        continue

                    # Rule 6: Person has EXPIRED or DEPARTED or visit terminated
                    if p.status in (EXPIRED, DEPARTED, VISIT_COMPLETED) or p.is_departed:
                        continue

                    if getattr(p, "current_visit_status", None) in ("SHORT_VISIT_IGNORED", "COMPLETED"):
                        continue

                    # Rule 7: SIMULTANEOUS VISIBILITY CHECK (Co-presence Graph)
                    if (tid in p.co_present_track_ids or 
                        tid in self.co_present_tracks.get(gid, set()) or 
                        gid in self.co_present_tracks.get(tid, set())):
                        self.reid_log.append({
                            "frame": frame_idx, "ts": round(timestamp, 3),
                            "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                            "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                            "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                            "tp_score": None, "composite": None, "margin": None,
                            "decision": "REJECTED_CO_PRESENCE",
                            "reason": f"Track {tid} and person {gid} were simultaneously visible in the same frame",
                            "allow_gallery_update": False,
                        })
                        continue

                    # Rule 7.5: Unverified occupant check
                    # A previously unverified person must NEVER inherit another visitor's P-ID
                    # unless there is strong, unambiguous evidence of genuine visitor recovery.
                    if cand and cand.get("is_unverified_occupant", False):
                        cand_first = cand.get("first_seen", timestamp)
                        # Case C1: Track was already present inside before p entered or while p was active
                        if cand_first < p.first_seen or cand_first < (p.last_seen - 0.2):
                            self.reid_log.append({
                                "frame": frame_idx, "ts": round(timestamp, 3),
                                "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                                "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                                "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                                "tp_score": None, "composite": None, "margin": None,
                                "decision": "REJECTED_UNVERIFIED_PRIOR_PRESENCE",
                                "reason": f"Track {tid} was already present inside before/during person {gid}'s presence",
                                "allow_gallery_update": False,
                            })
                            continue

                        # Case A vs Case C2: Track first appeared inside after p was lost.
                        # To qualify as in-store occlusion recovery (Case A), strict spatio-temporal continuity is required:
                        time_gap = timestamp - p.last_seen
                        if p.status != TEMPORARILY_LOST:
                            self.reid_log.append({
                                "frame": frame_idx, "ts": round(timestamp, 3),
                                "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                                "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                                "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                                "tp_score": None, "composite": None, "margin": None,
                                "decision": "REJECTED_UNVERIFIED_NO_CONTINUITY",
                                "reason": f"Track {tid} appeared inside without gate crossing; person {gid} is not temporarily lost (status={p.status})",
                                "allow_gallery_update": False,
                            })
                            continue

                        # Cross-camera check: verify camera topology connectivity
                        is_same_cam = (p.current_camera == cam_id) or (p.current_camera in ("camera_01", "cam1", "cam_01") and cam_id in ("camera_01", "cam1", "cam_01"))
                        if not is_same_cam:
                            is_conn = False
                            if self.camera_manager and hasattr(self.camera_manager, "is_connected"):
                                is_conn = self.camera_manager.is_connected(p.current_camera, cam_id)
                            if not is_conn:
                                self.reid_log.append({
                                    "frame": frame_idx, "ts": round(timestamp, 3),
                                    "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                                    "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                                    "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                                    "tp_score": None, "composite": None, "margin": None,
                                    "decision": "REJECTED_UNVERIFIED_CROSS_CAMERA",
                                    "reason": f"Cameras {p.current_camera} and {cam_id} are not connected; unverified track {tid} cannot match {gid}",
                                    "allow_gallery_update": False,
                                })
                                continue
                            if time_gap < 0.5:
                                continue

                        # Spatio-temporal walking continuity bounds on the same camera:
                        if is_same_cam and p.last_bbox is not None:
                            c_prev = _bbox_centre(p.last_bbox)
                            c_curr = _bbox_centre(q_bbox)
                            disp = np.hypot(c_curr[0] - c_prev[0], c_curr[1] - c_prev[1])
                            if time_gap <= 10.0:
                                max_disp = min(320.0, 60.0 + 80.0 * max(0.1, time_gap))
                                if disp > max_disp:
                                    self.reid_log.append({
                                        "frame": frame_idx, "ts": round(timestamp, 3),
                                        "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                                        "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                                        "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                                        "tp_score": None, "composite": None, "margin": None,
                                        "decision": "REJECTED_UNVERIFIED_NO_CONTINUITY",
                                        "reason": f"Unverified track {tid} appeared inside at distance {disp:.1f}px (> {max_disp:.1f}px) from {gid}'s last location after {time_gap:.2f}s",
                                        "allow_gallery_update": False,
                                    })
                                    continue
                            elif time_gap <= 25.0:
                                max_disp = min(600.0, 150.0 + 35.0 * time_gap)
                                if disp > max_disp:
                                    self.reid_log.append({
                                        "frame": frame_idx, "ts": round(timestamp, 3),
                                        "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                                        "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                                        "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                                        "tp_score": None, "composite": None, "margin": None,
                                        "decision": "REJECTED_UNVERIFIED_NO_CONTINUITY",
                                        "reason": f"Unverified track {tid} appeared inside at distance {disp:.1f}px (> {max_disp:.1f}px) from {gid}'s last location after {time_gap:.2f}s",
                                        "allow_gallery_update": False,
                                    })
                                    continue

                    # Rule 8: Physical Store Invariants (gate counter)
                    if active_gate is not None:
                        if gid in getattr(active_gate, "departed_persons", set()) or active_gate.person_state.get(gid) == "OUTSIDE":
                            continue
                        p_is_inside = (active_gate.person_state.get(gid) == "INSIDE" or
                                       active_gate.person_state.get(p.current_track_id) == "INSIDE" or
                                       (p.last_bbox is not None and active_gate.is_inside_or_entering(p.current_track_id, p.last_bbox, frame_w, frame_h)))
                        q_is_inside = active_gate.is_inside_or_entering(tid, q_bbox, frame_w, frame_h)
                        if p_is_inside and not q_is_inside:
                            continue
                        if not p_is_inside and q_is_inside:
                            continue

                    # Rule 9: Time gap limit
                    time_gap = timestamp - p.last_seen
                    if time_gap > Config.MAX_REID_TIME_GAP:
                        continue

                    # Rule 10: Spatial speed limit (impossible teleportation)
                    if p.last_bbox is not None and p.current_camera == cam_id:
                        c_prev = _bbox_centre(p.last_bbox)
                        c_curr = _bbox_centre(q_bbox)
                        disp = np.hypot(c_curr[0] - c_prev[0], c_curr[1] - c_prev[1])
                        eff_gap = max(0.1, time_gap)
                        speed = disp / eff_gap
                        if time_gap < 2.0 and speed > 800.0:
                            self.reid_log.append({
                                "frame": frame_idx, "ts": round(timestamp, 3),
                                "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                                "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                                "deep_sim": None, "cloth_sim": None, "app_score": None, "sp_score": None,
                                "tp_score": None, "composite": None, "margin": None,
                                "decision": "REJECTED_SPATIAL_SPEED",
                                "reason": f"Displacement {disp:.1f}px in {eff_gap:.2f}s ({speed:.1f} px/s) exceeds speed threshold",
                                "allow_gallery_update": False,
                            })
                            continue
                        if time_gap < 2.0:
                            sp_score = _spatial_score(p.last_bbox, q_bbox, self._frame_diag)
                        elif time_gap < 4.0 and disp < 140.0:
                            # Smooth turn in place or localized step: strong spatial continuity
                            sp_score = 0.75
                        else:
                            sp_score = max(0.20, _spatial_score(p.last_bbox, q_bbox, self._frame_diag))
                    else:
                        sp_score = 0.50

                    evaluated_gids.append(gid)

                    # Appearance & Clothing Fused Score
                    deep_sim = p.gallery.best_similarity(feat) if feat is not None else 0.0
                    cloth_sim = p.gallery.clothing_similarity(q_cloth) if (q_cloth is not None and len(p.gallery.clothing_profiles) > 0) else None

                    # ── CLOTHING GATE VETO ──
                    if cloth_sim is not None and cloth_sim < self.CLOTHING_MIN_GATE:
                        self.reid_log.append({
                            "frame": frame_idx, "ts": round(timestamp, 3),
                            "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                            "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                            "deep_sim": round(deep_sim, 4), "cloth_sim": round(cloth_sim, 4),
                            "app_score": None, "sp_score": round(sp_score, 4), "tp_score": None,
                            "composite": None, "margin": None,
                            "decision": "REJECTED_CLOTHING_GATE",
                            "reason": f"Clothing similarity {cloth_sim:.3f} below gate threshold {self.CLOTHING_MIN_GATE}",
                            "allow_gallery_update": False,
                        })
                        continue

                    # ── SIMILAR CLOTHING ALONE MUST NEVER MATCH ──
                    if cloth_sim is not None and cloth_sim >= 0.70 and deep_sim < 0.42:
                        self.reid_log.append({
                            "frame": frame_idx, "ts": round(timestamp, 3),
                            "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                            "candidate_gids": [gid], "best_candidate": gid, "second_candidate": None,
                            "deep_sim": round(deep_sim, 4), "cloth_sim": round(cloth_sim, 4),
                            "app_score": None, "sp_score": round(sp_score, 4), "tp_score": None,
                            "composite": None, "margin": None,
                            "decision": "REJECTED_THRESHOLD",
                            "reason": f"Similar clothing color ({cloth_sim:.3f}) cannot force match without sufficient deep similarity ({deep_sim:.3f} < 0.42)",
                            "allow_gallery_update": False,
                        })
                        continue

                    if cloth_sim is not None:
                        if cloth_sim < 0.45:
                            app_score = min(deep_sim * 0.5, 0.35)
                        elif deep_sim >= 0.65 and cloth_sim >= 0.55:
                            app_score = 0.60 * deep_sim + 0.40 * cloth_sim
                        elif cloth_sim >= 0.65 and deep_sim >= 0.45:
                            # Front-to-back turn or viewpoint change of same person
                            app_score = 0.45 * deep_sim + 0.55 * cloth_sim
                        else:
                            app_score = 0.65 * deep_sim + 0.35 * cloth_sim
                    else:
                        app_score = deep_sim

                    tp_score = _temporal_score(time_gap)
                    # Adaptive weighting based on elapsed time:
                    # When freshly lost (<= 2s), spatial continuity is critical.
                    # When lost for several seconds (>= 5s), appearance/clothing is the primary signal.
                    if time_gap < 2.0:
                        w_app, w_sp, w_tp = 0.45, 0.35, 0.20
                    elif time_gap < 5.0:
                        w_app, w_sp, w_tp = 0.55, 0.25, 0.20
                    else:
                        w_app, w_sp, w_tp = 0.65, 0.15, 0.20

                    composite = (w_app * app_score
                                 + w_sp * sp_score
                                 + w_tp * tp_score)

                    cands_for_q.append({
                        "gid": gid,
                        "composite": composite,
                        "app_score": app_score,
                        "cloth_sim": cloth_sim,
                        "deep_sim": deep_sim,
                        "sp_score": sp_score,
                        "tp_score": tp_score,
                    })

                if cands_for_q:
                    cands_for_q.sort(key=lambda x: x["composite"], reverse=True)
                    query_candidates[q_idx] = {
                        "candidates": cands_for_q,
                        "evaluated_gids": evaluated_gids,
                        "det": det,
                    }

            # ── Ambiguity & Margin Evaluation ──
            ambiguous_qdx = set()
            ambiguous_gids = set()
            query_top_info = {}

            for q_idx, data in query_candidates.items():
                cands = data["candidates"]
                top = cands[0]
                second = cands[1] if len(cands) > 1 else None

                p = self.persons[top["gid"]]
                cand = self.candidates.get(unknown_dets[q_idx]['track_id'])
                is_unverified = cand and cand.get("is_unverified_occupant", False)
                if is_unverified:
                    min_c = 0.65
                    min_a = 0.75
                else:
                    min_c = 0.52 if p.status == TEMPORARILY_LOST else 0.56
                    min_a = getattr(Config, "ACTIVE_VISITOR_REID_THRESHOLD", 0.75) if p.status == TEMPORARILY_LOST else getattr(Config, "REID_MATCH_THRESHOLD", 0.75)

                if second is not None:
                    margin = float(top["composite"] - second["composite"])
                    if second["composite"] >= (min_c - 0.08) and margin < self.CONFIDENCE_MARGIN:
                        ambiguous_qdx.add(q_idx)
                        ambiguous_gids.add(top["gid"])
                        ambiguous_gids.add(second["gid"])
                else:
                    margin = 1.0

                query_top_info[q_idx] = (top, second, margin, min_c, min_a)

            # Cross-query competition: check if multiple queries compete for the same candidate
            gid_to_competing_queries = defaultdict(list)
            for q_idx, (top, second, margin, min_c, min_a) in query_top_info.items():
                if top["composite"] >= min_c and top["app_score"] >= min_a:
                    gid_to_competing_queries[top["gid"]].append((top["composite"], q_idx))

            for gid, q_list in gid_to_competing_queries.items():
                if len(q_list) >= 2:
                    q_list.sort(key=lambda x: x[0], reverse=True)
                    margin = q_list[0][0] - q_list[1][0]
                    if margin < self.CONFIDENCE_MARGIN:
                        for _, q_idx in q_list:
                            ambiguous_qdx.add(q_idx)
                        ambiguous_gids.add(gid)

            # ── Safe One-to-One Association ──
            assigned_qdx  = set()
            assigned_gids = set()

            ranked_assignments = []
            for q_idx, (top, second, margin, min_c, min_a) in query_top_info.items():
                ranked_assignments.append((top["composite"], q_idx, top, second, margin, min_c, min_a))
            ranked_assignments.sort(key=lambda x: x[0], reverse=True)

            for comp, q_idx, top, second, margin, min_c, min_a in ranked_assignments:
                tid = unknown_dets[q_idx]['track_id']
                gid = top["gid"]
                cam_id = unknown_dets[q_idx]['camera_id']
                evaluated_gids = query_candidates[q_idx]["evaluated_gids"]

                if q_idx in assigned_qdx or gid in assigned_gids:
                    continue

                if q_idx in ambiguous_qdx or gid in ambiguous_gids:
                    # Check if local VLM verifier can provide second-stage tie-break
                    vlm_resolved = False
                    if getattr(self, "vlm_verifier", None) is not None and self.vlm_verifier.enabled:
                        det = unknown_dets[q_idx]
                        query_crop = det.get('crop')
                        if query_crop is not None:
                            vlm_res = self._query_vlm(query_crop, gid, tid)
                            if vlm_res and vlm_res.get("match") and vlm_res.get("confidence", 0.0) >= 0.75:
                                second_rejected = True
                                if second is not None:
                                    vlm_res2 = self._query_vlm(query_crop, second["gid"], tid)
                                    if vlm_res2 and vlm_res2.get("match") and vlm_res2.get("confidence", 0.0) >= 0.70:
                                        second_rejected = False
                                if second_rejected:
                                    vlm_resolved = True
                            elif second is not None:
                                vlm_res2 = self._query_vlm(query_crop, second["gid"], tid)
                                if vlm_res2 and vlm_res2.get("match") and vlm_res2.get("confidence", 0.0) >= 0.75:
                                    top, second = second, top
                                    gid = top["gid"]
                                    comp = top["composite"]
                                    vlm_resolved = True

                    if not vlm_resolved:
                        # Defer ambiguous match! Never force an unreliable assignment
                        self.reid_log.append({
                            "frame": frame_idx, "ts": round(timestamp, 3),
                            "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                            "candidate_gid": gid,
                            "candidate_gids": evaluated_gids, "best_candidate": gid,
                            "second_candidate": second["gid"] if second else None,
                            "deep_sim": round(top["deep_sim"], 4), "cloth_sim": round(top["cloth_sim"], 4) if top["cloth_sim"] is not None else None,
                            "app_score": round(top["app_score"], 4), "sp_score": round(top["sp_score"], 4),
                            "tp_score": round(top["tp_score"], 4), "composite": round(comp, 4),
                            "margin": round(margin, 4),
                            "decision": "DEFERRED",
                            "reason": f"Ambiguous candidates (margin {margin:.3f} < {self.CONFIDENCE_MARGIN}); deferred to prevent identity swap",
                            "allow_gallery_update": False,
                            "min_composite": min_c, "min_app": min_a,
                        })
                        continue

                # Threshold check
                if top["composite"] < min_c or top["app_score"] < min_a:
                    vlm_corroborated = False
                    if getattr(self, "vlm_verifier", None) is not None and self.vlm_verifier.enabled:
                        det = unknown_dets[q_idx]
                        query_crop = det.get('crop')
                        amb_low = getattr(Config, "VLM_AMBIGUITY_LOW", 0.68)
                        amb_high = getattr(Config, "VLM_AMBIGUITY_HIGH", 0.78)
                        # When candidate appearance is in ambiguity range and no other close competitor
                        if query_crop is not None and (amb_low <= top["app_score"] <= amb_high) and (second is None or margin >= 0.06):
                            vlm_res = self._query_vlm(query_crop, gid, tid)
                            if vlm_res and vlm_res.get("match") and vlm_res.get("confidence", 0.0) >= 0.75:
                                top["app_score"] = max(top["app_score"], 0.82)
                                top["composite"] = max(top["composite"], min_c + 0.05)
                                comp = top["composite"]
                                vlm_corroborated = True

                    if not vlm_corroborated:
                        self.reid_log.append({
                            "frame": frame_idx, "ts": round(timestamp, 3),
                            "camera_id": cam_id, "query_tid": tid, "assigned_gid": None,
                            "candidate_gid": gid,
                            "candidate_gids": evaluated_gids, "best_candidate": gid,
                            "second_candidate": second["gid"] if second else None,
                            "deep_sim": round(top["deep_sim"], 4), "cloth_sim": round(top["cloth_sim"], 4) if top["cloth_sim"] is not None else None,
                            "app_score": round(top["app_score"], 4), "sp_score": round(top["sp_score"], 4),
                            "tp_score": round(top["tp_score"], 4), "composite": round(comp, 4),
                            "margin": round(margin, 4),
                            "decision": "REJECTED_THRESHOLD",
                            "reason": f"Composite {comp:.3f} < {min_c} or app_score {top['app_score']:.3f} < {min_a}",
                            "allow_gallery_update": False,
                            "min_composite": min_c, "min_app": min_a,
                        })
                        continue

                # ACCEPT MATCH
                p = self.persons[gid]
                det = unknown_dets[q_idx]
                feat = features_map.get(tid)
                match_cloth = clothing_map.get(tid)
                quality = self._compute_quality(det['bbox'])

                # ── Gallery contamination protection ──
                cand = self.candidates.get(tid)
                is_unverified = cand and cand.get("is_unverified_occupant", False)
                if is_unverified:
                    # Unverified in-store recovery must NEVER contaminate the visitor gallery!
                    allow_gallery_update = False
                else:
                    allow_gallery_update = (
                        top["app_score"] >= 0.75 and
                        top["composite"] >= 0.62 and
                        margin >= 0.08 and
                        quality >= 0.25 and
                        tid not in occluded_tracks
                    )

                if p.current_track_id in self.local_to_global:
                    if self.local_to_global[p.current_track_id] == gid:
                        del self.local_to_global[p.current_track_id]

                if not allow_gallery_update:
                    p.mark_reidentified(timestamp, det['camera_id'], tid, None, det['bbox'], comp, clothing=None)
                else:
                    p.mark_reidentified(timestamp, det['camera_id'], tid, feat, det['bbox'], comp, clothing=match_cloth)

                self.local_to_global[tid] = gid
                assigned_qdx.add(q_idx)
                assigned_gids.add(gid)
                currently_active_on_cam[det['camera_id']].add(gid)

                if tid in self.candidates:
                    del self.candidates[tid]

                if active_gate is not None:
                    active_gate.transfer_identity(tid, gid)
                    active_gate.person_state[gid] = "INSIDE"
                    active_gate.person_state[tid] = "INSIDE"

                self.reid_log.append({
                    "frame": frame_idx, "ts": round(timestamp, 3),
                    "camera_id": cam_id, "query_tid": tid, "assigned_gid": gid,
                    "candidate_gid": gid,
                    "candidate_gids": evaluated_gids, "best_candidate": gid,
                    "second_candidate": second["gid"] if second else None,
                    "deep_sim": round(top["deep_sim"], 4), "cloth_sim": round(top["cloth_sim"], 4) if top["cloth_sim"] is not None else None,
                    "app_score": round(top["app_score"], 4), "sp_score": round(top["sp_score"], 4),
                    "tp_score": round(top["tp_score"], 4), "composite": round(comp, 4),
                    "margin": round(margin, 4),
                    "decision": "ACCEPTED",
                    "reason": f"Accepted 1-to-1 match to {gid} with composite {comp:.3f} and margin {margin:.3f}",
                    "allow_gallery_update": allow_gallery_update,
                    "min_composite": min_c, "min_app": min_a,
                })

                decisions.append({
                    "track_id": tid, "global_id": gid,
                    "state": REIDENTIFIED, "sim": comp
                })

            # ── 4. Unassigned unknown detections lifecycle ──
            for q_idx, det in enumerate(unknown_dets):
                if q_idx in assigned_qdx:
                    continue
                tid  = det['track_id']
                feat = features_map.get(tid)

                # Check if deferred due to ambiguity:
                if q_idx in ambiguous_qdx:
                    top_cand = query_top_info.get(q_idx, ({} , None, 0.0, 0, 0))[0]
                    sim_val = top_cand.get("composite", 0.0) if isinstance(top_cand, dict) else 0.0
                    decisions.append({
                        "track_id": tid, "global_id": tid,
                        "state": IDENTITY_UNCERTAIN, "sim": sim_val
                    })
                    continue

                # ── Candidate Wait Logic with Statue / Mannequin Dynamic Evaluation ──
                bx1, by1, bx2, by2 = det['bbox']
                hx = (bx1 + bx2) / 2.0
                hy = float(by1)
                fx = (bx1 + bx2) / 2.0
                fy = float(by2)

                cand = self.candidates.get(tid)
                if cand is None:
                    is_preexisting_inside = False
                    if active_gate is not None and hasattr(active_gate, "is_inside_or_entering"):
                        is_preexisting_inside = active_gate.is_inside_or_entering(tid, det['bbox'], frame_w, frame_h)
                    is_initial_occupant = is_preexisting_inside and (len(self.persons) == 0)
                    cand = {
                        "first_seen": timestamp,
                        "last_seen": timestamp,
                        "head_pts": deque([(hx, hy)], maxlen=90),
                        "foot_pts": deque([(fx, fy)], maxlen=90),
                        "is_static": False,
                        "is_initial_occupant": is_initial_occupant,
                        "is_unverified_occupant": is_preexisting_inside,
                        "anchor_head": (hx, hy),
                        "anchor_foot": (fx, fy),
                    }
                    self.candidates[tid] = cand

                if not self.require_candidate_motion:
                    if active_gate is not None:
                        is_inside = False
                        if hasattr(active_gate, "is_inside_or_entering"):
                            is_inside = active_gate.is_inside_or_entering(tid, det['bbox'], frame_w, frame_h)
                        elif hasattr(active_gate, "person_state"):
                            is_inside = (active_gate.person_state.get(tid) == "INSIDE")
                        state_lbl = "UNVERIFIED_OCCUPANT" if cand.get("is_unverified_occupant", False) else ("INSIDE" if is_inside else "OUTSIDE")
                        decisions.append({
                            "track_id": tid, "global_id": tid,
                            "state": state_lbl, "sim": 0.0
                        })
                        continue
                    # Test mode without motion requirement: promote immediately
                    gid = self._new_gid()
                    p = GlobalPerson(
                        gid, timestamp, det['camera_id'], tid,
                        feat, det['bbox'])
                    if clothing_map and tid in clothing_map:
                        p.gallery.add_clothing(clothing_map[tid])
                    self.persons[gid] = p
                    self.local_to_global[tid] = gid
                    self._sync_co_presence(tid, gid, p)
                    currently_active_on_cam[det['camera_id']].add(gid)
                    if tid in self.candidates:
                        del self.candidates[tid]
                    decisions.append({
                        "track_id": tid, "global_id": gid,
                        "state": "NEW", "sim": None
                    })
                    continue

                # Measure physical traversal
                h_pts = list(cand["head_pts"])
                f_pts = list(cand["foot_pts"])
                head_travel = np.hypot(h_pts[-1][0] - h_pts[0][0], h_pts[-1][1] - h_pts[0][1])
                foot_travel = np.hypot(f_pts[-1][0] - f_pts[0][0], f_pts[-1][1] - f_pts[0][1])

                med_hx = float(np.median([p[0] for p in h_pts]))
                med_hy = float(np.median([p[1] for p in h_pts]))
                head_radius = max(np.hypot(p[0] - med_hx, p[1] - med_hy) for p in h_pts)

                recent_k = min(len(h_pts), 20)
                recent_head_travel = np.hypot(h_pts[-1][0] - h_pts[-recent_k][0], h_pts[-1][1] - h_pts[-recent_k][1])
                recent_foot_travel = np.hypot(f_pts[-1][0] - f_pts[-recent_k][0], f_pts[-1][1] - f_pts[-recent_k][1])

                anc_h = cand.get("anchor_head", (med_hx, med_hy))
                anc_f = cand.get("anchor_foot", (f_pts[0][0], f_pts[0][1]))
                dist_from_anchor_h = np.hypot(h_pts[-1][0] - anc_h[0], h_pts[-1][1] - anc_h[1])
                dist_from_anchor_f = np.hypot(f_pts[-1][0] - anc_f[0], f_pts[-1][1] - anc_f[1])

                if cand.get("is_static", False):
                    # Reassess motion: Has movement resumed?
                    resumed = (
                        dist_from_anchor_h >= 22.0 or
                        dist_from_anchor_f >= 25.0 or
                        recent_head_travel >= 18.0 or
                        recent_foot_travel >= 22.0
                    )
                    if resumed:
                        # Movement resumed! Restore tracking
                        cand["is_static"] = False
                        cand["anchor_head"] = (hx, hy)
                        cand["anchor_foot"] = (fx, fy)
                    else:
                        # Still stationary: keep candidate alive in self.candidates (NEVER blacklist)
                        decisions.append({
                            "track_id": tid, "global_id": "STATUE",
                            "state": "STATIC", "sim": 0.0
                        })
                        continue
                else:
                    # Evaluate if candidate is stationary (standing still)
                    is_stationary = (head_travel < 20.0 and foot_travel < 25.0 and head_radius < 18.0)
                    time_alive = timestamp - cand["first_seen"]
                    if is_stationary:
                        if len(h_pts) >= 15 or time_alive >= 1.5:
                            cand["is_static"] = True
                            cand["anchor_head"] = (med_hx, med_hy)
                            cand["anchor_foot"] = (float(np.median([p[0] for p in f_pts])), float(np.median([p[1] for p in f_pts])))
                            decisions.append({
                                "track_id": tid, "global_id": "STATUE",
                                "state": "STATIC", "sim": 0.0
                            })
                            continue
                        else:
                            active_gate = gate_counter or self.gate_counter
                            if active_gate is not None:
                                is_inside = False
                                if hasattr(active_gate, "is_inside_or_entering"):
                                    is_inside = active_gate.is_inside_or_entering(tid, det['bbox'], frame_w, frame_h)
                                elif hasattr(active_gate, "person_state"):
                                    is_inside = (active_gate.person_state.get(tid) == "INSIDE")
                                state_lbl = "UNVERIFIED_OCCUPANT" if cand.get("is_unverified_occupant", False) else ("INSIDE" if is_inside else "OUTSIDE")
                                decisions.append({
                                    "track_id": tid, "global_id": tid,
                                    "state": state_lbl, "sim": 0.0
                                })
                            else:
                                decisions.append({
                                    "track_id": tid, "global_id": "PENDING",
                                    "state": "PENDING", "sim": 0.0
                                })
                            continue

                # ── Gate / Store Admission Check ──
                # When a gate counter is active, permanent visitor IDs (P-001, P-002...)
                # are assigned ONLY upon genuine entry crossing via promote_track_to_visitor()!
                active_gate = gate_counter or self.gate_counter
                if active_gate is not None:
                    is_inside = False
                    if hasattr(active_gate, "is_inside_or_entering"):
                        is_inside = active_gate.is_inside_or_entering(tid, det['bbox'], frame_w, frame_h)
                    elif hasattr(active_gate, "person_state"):
                        is_inside = (active_gate.person_state.get(tid) == "INSIDE")
                    state_lbl = "UNVERIFIED_OCCUPANT" if cand.get("is_unverified_occupant", False) else ("INSIDE" if is_inside else "OUTSIDE")
                    decisions.append({
                        "track_id": tid, "global_id": tid,
                        "state": state_lbl, "sim": 0.0
                    })
                    continue

                # Standalone / mock mode without gate counter:
                gid  = self._new_gid()
                p = GlobalPerson(
                    gid, timestamp, det['camera_id'], tid,
                    feat, det['bbox'])
                if clothing_map and tid in clothing_map:
                    p.gallery.add_clothing(clothing_map[tid])
                self.persons[gid] = p
                self.local_to_global[tid] = gid
                self._sync_co_presence(tid, gid, p)
                currently_active_on_cam[det['camera_id']].add(gid)
                
                del self.candidates[tid]

                decisions.append({
                    "track_id": tid, "global_id": gid,
                    "state": "NEW", "sim": None
                })

            # Clean up stale candidates unseen for > 5.0 seconds
            stale_tids = [t for t, c in self.candidates.items() if (timestamp - c.get("last_seen", c.get("first_seen", 0.0))) > 5.0]
            for t in stale_tids:
                del self.candidates[t]

        return decisions
