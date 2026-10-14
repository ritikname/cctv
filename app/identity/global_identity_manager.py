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
ACTIVE           = "ACTIVE"
TEMPORARILY_LOST = "TEMPORARILY_LOST"
REIDENTIFIED     = "REIDENTIFIED"
EXPIRED          = "EXPIRED"
DEPARTED         = "DEPARTED"


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

            # Evaluate rolling motion over the last 40-60 frames (1.5 - 2.0 seconds):
            if len(self.head_history) >= 40:
                h_pts = list(self.head_history)
                f_pts = list(self.foot_history)

                # Head trajectory: A statue's head is completely fixed on a stand/neck!
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

                # If head and feet barely move across 40+ frames, they are temporarily stationary (paused)
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

                    # 1. High deep similarity -> safely add to corresponding view.
                    if sim_check >= 0.60:
                        self.gallery.add(embedding, view_tag=view_tag, clothing=clothing)
                    # 2. Front-to-back turn during continuous tracking:
                    # When a person turns, deep sim may drop to 0.35-0.60, but
                    # clothing matches strongly (>= 0.52). Update multi-view gallery and clothing profile.
                    elif cloth_check is not None and cloth_check >= 0.52 and sim_check >= 0.35:
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

    def _new_gid(self):
        gid = f"P-{self._next_idx:03d}"
        self._next_idx += 1
        return gid

    def get_reid_log(self, decision_filter=None):
        """Return the Re-ID audit log, optionally filtered by decision type.
        
        decision_filter: None (all), 'ACCEPTED', 'REJECTED_THRESHOLD',
                         'REJECTED_CLOTHING_GATE', or 'CANDIDATE'
        """
        if decision_filter is None:
            return list(self.reid_log)
        return [e for e in self.reid_log if e.get("decision") == decision_filter]

    def dump_reid_log(self, path: str):
        """Write the full Re-ID audit log to a CSV file for external analysis."""
        import csv, os
        if not self.reid_log:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = [
            "frame", "ts", "query_tid", "candidate_gid",
            "deep_sim", "cloth_sim", "app_score", "sp_score",
            "tp_score", "composite", "allow_gallery_update", "decision",
            "min_composite", "min_app",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
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

    def promote_track_to_visitor(self, tid, timestamp, cam_id="camera_01", feat=None, bbox=None, clothing=None):
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

        if feat is None:
            feat = self.latest_features.get(tid)
        if clothing is None:
            clothing = self.latest_clothing.get(tid)

        # Check if this person is a returning visitor who previously completed a visit
        # within the configured retention window:
        departed_match = None
        best_sim = 0.0
        if feat is not None:
            retention_limit = getattr(Config, "VISITOR_RETENTION_TIME", 3600)
            strong_thresh = getattr(Config, "REID_STRONG_MATCH_THRESHOLD", 0.75)
            for gid, p in self.persons.items():
                if p.is_departed and p.current_visit_status == "COMPLETED":
                    if (timestamp - p.last_seen) <= retention_limit:
                        sim = p.gallery.best_similarity(feat)
                        if clothing and len(p.gallery.clothing_profiles) > 0:
                            c_sim = p.gallery.clothing_similarity(clothing)
                            sim = 0.70 * sim + 0.30 * c_sim
                        if sim >= strong_thresh and sim > best_sim:
                            best_sim = sim
                            departed_match = p

        if departed_match is not None:
            # Genuine re-entry of previously known visitor! Start a new separate visit record:
            departed_match.start_new_visit(timestamp, cam_id, tid, feat, bbox, clothing=clothing)
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
                if p and p.is_departed:
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
                # Check if this detection is already inside when first appearing
                is_preexisting_inside = False
                if active_gate is not None and hasattr(active_gate, "is_inside_or_entering"):
                    is_preexisting_inside = active_gate.is_inside_or_entering(tid, det['bbox'], frame_w, frame_h)

                # An initial occupant was present before any visitor entries occurred
                is_initial_occupant = is_preexisting_inside and (len(self.persons) == 0)

                self.candidates[tid] = {
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "head_pts": deque(maxlen=90),
                    "foot_pts": deque(maxlen=90),
                    "is_static": False,
                    "is_initial_occupant": is_initial_occupant,
                    "is_unverified_occupant": is_preexisting_inside,
                    "anchor_head": (hx, hy),
                    "anchor_foot": (fx, fy),
                }
            cand = self.candidates[tid]
            cand["last_seen"] = timestamp
            cand["head_pts"].append((hx, hy))
            cand["foot_pts"].append((fx, fy))

        # ── 3. Re-ID unknown tracks (COMPOSITE scoring) ──────────────────
        if unknown_dets:
            currently_active_on_cam = defaultdict(set)
            for det in known_dets:
                tid = det['track_id']
                gid = self.local_to_global[tid]
                currently_active_on_cam[det['camera_id']].add(gid)

            all_candidates = []
            for q_idx, det in enumerate(unknown_dets):
                tid     = det['track_id']
                cam_id  = det['camera_id']
                feat    = features_map.get(tid)
                q_bbox  = det['bbox']
                
                # Rule 1: Static candidate (statue / mannequin) can NEVER be Re-ID matched!
                cand = self.candidates.get(tid)
                if cand and cand.get("is_static", False):
                    all_candidates.append([])
                    continue

                # Rule 2: Initial staff/occupant present before entries began can NEVER be Re-ID matched!
                if cand and cand.get("is_initial_occupant", False):
                    all_candidates.append([])
                    continue

                # Rule 3: Do not attempt Re-ID with poor quality crop
                quality = self._compute_quality(q_bbox)
                if feat is None or quality < 0.20:
                    all_candidates.append([])
                    continue

                row = []
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
                    if p.status in (EXPIRED, DEPARTED) or p.is_departed:
                        continue

                    if getattr(p, "current_visit_status", None) in ("SHORT_VISIT_IGNORED", "COMPLETED"):
                        continue

                    # Rule 7: SIMULTANEOUS VISIBILITY CHECK
                    # If tid and gid (or p) were ever visible at the same time, REJECT!
                    if (tid in p.co_present_track_ids or 
                        tid in self.co_present_tracks.get(gid, set()) or 
                        gid in self.co_present_tracks.get(tid, set())):
                        continue

                    # Rule 8: Physical Store Invariants (gate counter)
                    if active_gate is not None:
                        # Exited or departed person cannot be inside
                        if gid in getattr(active_gate, "departed_persons", set()) or active_gate.person_state.get(gid) == "OUTSIDE":
                            continue
                        p_is_inside = (active_gate.person_state.get(gid) == "INSIDE" or
                                       active_gate.person_state.get(p.current_track_id) == "INSIDE")
                        q_is_inside = active_gate.is_inside_or_entering(tid, q_bbox, frame_w, frame_h)
                        # A person currently inside cannot suddenly appear outside
                        if p_is_inside and not q_is_inside:
                            continue
                        # A person outside cannot match a track deep inside without crossing
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
                            continue
                        if time_gap < 2.0:
                            sp_score = _spatial_score(p.last_bbox, q_bbox, self._frame_diag)
                        elif time_gap < 4.0 and disp < 140.0:
                            # Smooth turn in place or localized step: strong spatial continuity
                            sp_score = 0.75
                        else:
                            sp_score = 0.5
                    else:
                        sp_score = 0.5

                    # Appearance & Clothing Fused Score
                    deep_sim = p.gallery.best_similarity(feat) if feat is not None else 0.0
                    q_cloth = clothing_map.get(tid)
                    cloth_sim = p.gallery.clothing_similarity(q_cloth) if (q_cloth is not None and len(p.gallery.clothing_profiles) > 0) else None

                    # ── CLOTHING GATE VETO ────────────────────────────────────
                    # If a clothing profile exists for the gallery person AND the
                    # query, reject the match outright if clothing similarity is
                    # below the gate threshold. This is the primary defence
                    # against cross-person identity transfer.
                    clothing_gate_ok = True
                    if cloth_sim is not None:
                        if cloth_sim < self.CLOTHING_MIN_GATE:
                            clothing_gate_ok = False

                    if not clothing_gate_ok:
                        # Log the rejection for auditability
                        self.reid_log.append({
                            "frame": frame_idx, "ts": round(timestamp, 3),
                            "query_tid": tid, "candidate_gid": gid,
                            "deep_sim": round(deep_sim, 4),
                            "cloth_sim": round(cloth_sim, 4) if cloth_sim is not None else None,
                            "decision": "REJECTED_CLOTHING_GATE",
                        })
                        continue

                    if cloth_sim is not None:
                        if cloth_sim < 0.40:
                            # Significant clothing mismatch: penalise appearance score
                            app_score = min(deep_sim * 0.5, 0.35)
                        elif deep_sim >= 0.65 and cloth_sim >= 0.55:
                            # Both deep feature and clothing match strongly
                            app_score = max(deep_sim, 0.60 * deep_sim + 0.40 * cloth_sim)
                        elif cloth_sim >= 0.48 and deep_sim >= 0.35:
                            # Front-to-back turn or viewpoint change of same person:
                            # Clothing colors provide robust view-invariant signal.
                            # Combined confidence scales so genuine returning visitors recover.
                            app_score = max(deep_sim, min(1.0, 0.45 * deep_sim + 0.55 * cloth_sim + 0.26))
                        else:
                            app_score = 0.60 * deep_sim + 0.40 * cloth_sim
                    else:
                        app_score = deep_sim

                    tp_score = _temporal_score(time_gap)

                    composite = (self.W_APPEARANCE * app_score
                                 + self.W_SPATIAL * sp_score
                                 + self.W_TEMPORAL * tp_score)

                    # Log candidate match for audit
                    self.reid_log.append({
                        "frame": frame_idx, "ts": round(timestamp, 3),
                        "query_tid": tid, "candidate_gid": gid,
                        "deep_sim": round(deep_sim, 4),
                        "cloth_sim": round(cloth_sim, 4) if cloth_sim is not None else None,
                        "app_score": round(app_score, 4),
                        "sp_score": round(sp_score, 4),
                        "tp_score": round(tp_score, 4),
                        "composite": round(composite, 4),
                        "decision": "CANDIDATE",
                    })

                    row.append((composite, app_score, cloth_sim, gid))

                all_candidates.append(row)

            # Greedy assignment: highest composite first
            flat = []
            for q_idx, row in enumerate(all_candidates):
                for composite, app_score, cloth_sim, gid in row:
                    flat.append((composite, app_score, cloth_sim, q_idx, gid))
            flat.sort(reverse=True)

            # Safe Ambiguity Detection (Requirement 4):
            # Defer match if multiple candidates compete with near-identical composite scores
            ambiguous_qdx  = set()
            ambiguous_gids = set()

            q_matches = defaultdict(list)
            g_matches = defaultdict(list)
            for composite, app_score, cloth_sim, q_idx, gid in flat:
                p = self.persons[gid]
                min_c = 0.52 if p.status == TEMPORARILY_LOST else 0.56
                min_a = getattr(Config, "ACTIVE_VISITOR_REID_THRESHOLD", 0.75) if p.status == TEMPORARILY_LOST else getattr(Config, "REID_MATCH_THRESHOLD", 0.75)
                if composite >= min_c and app_score >= min_a:
                    q_matches[q_idx].append(composite)
                    g_matches[gid].append(composite)

            for q_idx, comps in q_matches.items():
                if len(comps) >= 2:
                    comps.sort(reverse=True)
                    if (comps[0] - comps[1]) < 0.035:
                        ambiguous_qdx.add(q_idx)

            for gid, comps in g_matches.items():
                if len(comps) >= 2:
                    comps.sort(reverse=True)
                    if (comps[0] - comps[1]) < 0.035:
                        ambiguous_gids.add(gid)

            assigned_qdx  = set()
            assigned_gids = set()

            for composite, app_score, cloth_sim, q_idx, gid in flat:
                if q_idx in assigned_qdx or gid in assigned_gids:
                    continue

                # Ambiguity protection: if candidates are ambiguous, defer rather than force!
                if q_idx in ambiguous_qdx or gid in ambiguous_gids:
                    continue

                det = unknown_dets[q_idx]
                cam_id = det['camera_id']
                p = self.persons[gid]
                candidate_time_gap = timestamp - p.last_seen

                min_composite = 0.56
                min_app = getattr(Config, "REID_MATCH_THRESHOLD", 0.75)

                # For active registered visitors recovering after temporary disappearance:
                if p.status == TEMPORARILY_LOST:
                    min_composite = 0.52
                    min_app = getattr(Config, "ACTIVE_VISITOR_REID_THRESHOLD", 0.75)
                elif p.current_camera == cam_id and candidate_time_gap <= 20.0:
                    min_composite = 0.56
                    min_app = getattr(Config, "REID_MATCH_THRESHOLD", 0.75)

                # Reject uncertain matches!
                if composite < min_composite or app_score < min_app:
                    # Log final rejection
                    self.reid_log.append({
                        "frame": frame_idx, "ts": round(timestamp, 3),
                        "query_tid": det['track_id'], "candidate_gid": gid,
                        "composite": round(composite, 4), "app_score": round(app_score, 4),
                        "cloth_sim": round(cloth_sim, 4) if cloth_sim is not None else None,
                        "decision": "REJECTED_THRESHOLD",
                        "min_composite": min_composite, "min_app": min_app,
                    })
                    continue

                tid  = det['track_id']
                feat = features_map.get(tid)
                match_cloth = clothing_map.get(tid)

                # ── Gallery contamination protection ──────────────────────────
                # Only update the gallery with this new embedding if the match
                # is high-quality. Low-confidence matches (barely above threshold)
                # must NOT pollute the gallery with a potentially wrong person's
                # features. This prevents a wrong match from cascading and causing
                # all future matches to also be wrong.
                allow_gallery_update = (
                    app_score >= 0.75 and composite >= 0.60
                )

                if p.current_track_id in self.local_to_global:
                    if self.local_to_global[p.current_track_id] == gid:
                        del self.local_to_global[p.current_track_id]

                # Mark re-identified; gallery update gated by quality
                if not allow_gallery_update:
                    # Suppress embedding update: pass None for feat and clothing
                    p.mark_reidentified(
                        timestamp, det['camera_id'], tid, None,
                        det['bbox'], composite, clothing=None)
                else:
                    p.mark_reidentified(
                        timestamp, det['camera_id'], tid, feat,
                        det['bbox'], composite, clothing=match_cloth)

                self.local_to_global[tid] = gid
                assigned_qdx.add(q_idx)
                assigned_gids.add(gid)
                currently_active_on_cam[det['camera_id']].add(gid)

                # Log accepted match
                self.reid_log.append({
                    "frame": frame_idx, "ts": round(timestamp, 3),
                    "query_tid": tid, "candidate_gid": gid,
                    "composite": round(composite, 4), "app_score": round(app_score, 4),
                    "cloth_sim": round(cloth_sim, 4) if cloth_sim is not None else None,
                    "allow_gallery_update": allow_gallery_update,
                    "decision": "ACCEPTED",
                })

                if tid in self.candidates:
                    del self.candidates[tid]

                if active_gate is not None:
                    active_gate.transfer_identity(tid, gid)
                    active_gate.person_state[gid] = "INSIDE"
                    active_gate.person_state[tid] = "INSIDE"

                decisions.append({
                    "track_id": tid, "global_id": gid,
                    "state": REIDENTIFIED, "sim": composite
                })

            # Remaining → new persons
            for q_idx, det in enumerate(unknown_dets):
                if q_idx in assigned_qdx:
                    continue
                tid  = det['track_id']
                feat = features_map.get(tid)
                
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
                    if active_gate is not None and cand.get("is_unverified_occupant", False):
                        decisions.append({
                            "track_id": tid, "global_id": tid,
                            "state": "UNVERIFIED_OCCUPANT", "sim": 0.0
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
                    # A walking human travels > 30-100 pixels in 2-3 seconds.
                    # A statue / standing person has head_travel < 20, foot_travel < 25, head_radius < 18
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
                # Temporary tracks (approaching, outside, or already inside) remain as temporary tracking IDs.
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

        # ── 4. Mark lost ──────────────────────────────────────────────────
        for gid, p in self.persons.items():
            if p.status in (ACTIVE, REIDENTIFIED):
                if p.current_track_id not in active_tracks:
                    p.mark_lost()

        # ── 5. Expire ────────────────────────────────────────────────────
        for gid, p in self.persons.items():
            if p.status == TEMPORARILY_LOST:
                if (timestamp - p.last_seen) > Config.MAX_REID_TIME_GAP:
                    p.mark_expired()

        return decisions
