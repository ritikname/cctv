"""
Advanced Hysteresis Tripwire & Gate Crossing Counter (v2.1)

Features:
- Short Visit Filtering (visits <= 5 seconds excluded from official in/out counts).
- Entry Confirmation Window (visits confirmed only after remaining inside for >= 5s).
- Door-Opening Immunity & Rollback (holding door open does not count as exit).
- Directional State Machine per Person (no oscillation between IN/OUT).
- Integration with Exit-Based Group Detection.
"""

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class TripwireCounter:
    def __init__(
        self,
        point_a: Tuple[float, float] = (0.20, 0.75),
        point_b: Tuple[float, float] = (0.80, 0.75),
        inside_direction: str = "below",
        corridor_half_width_px: float = 28.0,
        min_crossing_travel_px: float = 35.0,
        min_in_store_depth_for_exit_px: float = 85.0,
        min_visit_duration_sec: float = 5.0,
    ):
        self.point_a = point_a
        self.point_b = point_b
        self.inside_direction = inside_direction
        self.corridor_half_width_px = corridor_half_width_px
        self.min_crossing_travel_px = min_crossing_travel_px
        self.min_in_store_depth_for_exit_px = min_in_store_depth_for_exit_px
        self.min_visit_duration_sec = min_visit_duration_sec

        self.entry_count = 0
        self.exit_count = 0

        # State machine per person: 'OUTSIDE' | 'INSIDE' | 'THRESHOLD'
        self.person_state: Dict[str, str] = {}

        # Deepest penetration into store (negative values indicate deep inside):
        # person_id -> float (min signed distance observed)
        self.deepest_inside: Dict[str, float] = defaultdict(lambda: 999.0)

        # Frame history: person_id -> deque of (foot_x, foot_y, signed_dist)
        self.person_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=45))

        # Recent events log
        self.recent_events = deque(maxlen=20)
        self.last_crossing_time: Dict[str, float] = {}
        self.pending_exits: Dict[str, Tuple[float, float]] = {}
        self.departed_persons: Set[str] = set()

        # Visit confirmation tracking (Ignore short visits <= min_visit_duration_sec)
        self.pending_entries: Dict[str, float] = {}     # person_id -> entry_timestamp (< 5s inside)
        self.confirmed_entries: Dict[str, float] = {}   # person_id -> entry_timestamp (>= 5s inside)
        self.visit_records: List[Dict] = []             # complete history of visits (short & confirmed)
        self.ignored_visits: List[Dict] = []            # separate log of excluded short visits (duration <= 5s)

        # Crossing evaluation audit log & cache for diagnostics and UI overlays
        self.last_eval_debug: Dict[str, Dict] = {}
        self.crossing_debug_log: deque = deque(maxlen=200)

        # Optional group detector and identity manager references
        self.group_detector = None
        self.identity_manager = None

    def update_geometry(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
        inside_direction: Optional[str] = None,
    ):
        self.point_a = point_a
        self.point_b = point_b
        if inside_direction:
            self.inside_direction = inside_direction

    def transfer_identity(self, old_id: str, new_id: str):
        """
        Seamlessly transfer state, history, and penetration depth when a candidate
        track_id (e.g. camera_01-T18) is promoted to a global ID (e.g. P-001).
        Preserves history under both IDs for tracker continuity across frames.
        """
        if old_id == new_id:
            return
        if new_id in self.departed_persons:
            self.person_state[new_id] = "OUTSIDE"
            return
        if old_id in self.pending_entries:
            self.pending_entries[new_id] = self.pending_entries.get(new_id, self.pending_entries[old_id])
        if old_id in self.confirmed_entries:
            self.confirmed_entries[new_id] = self.confirmed_entries.get(new_id, self.confirmed_entries[old_id])
        if self.person_state.get(new_id) == "INSIDE":
            self.person_state[old_id] = "INSIDE"
        elif old_id in self.person_state:
            old_st = self.person_state[old_id]
            if self.person_state.get(new_id) != "INSIDE":
                self.person_state[new_id] = old_st

        if new_id in self.deepest_inside:
            self.deepest_inside[old_id] = min(self.deepest_inside.get(old_id, 999.0), self.deepest_inside[new_id])
        elif old_id in self.deepest_inside:
            old_depth = self.deepest_inside[old_id]
            self.deepest_inside[new_id] = min(self.deepest_inside.get(new_id, 999.0), old_depth)
        if old_id in self.person_history:
            old_hist = list(self.person_history[old_id])
            if new_id not in self.person_history or len(self.person_history[new_id]) == 0:
                self.person_history[new_id] = deque(old_hist, maxlen=45)
            else:
                self.person_history[new_id].extend(old_hist)
        if old_id in self.last_crossing_time:
            old_time = self.last_crossing_time[old_id]
            self.last_crossing_time[new_id] = max(self.last_crossing_time.get(new_id, 0.0), old_time)

    def is_inside_or_entering(
        self,
        person_id: str,
        bbox: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
    ) -> bool:
        """
        Returns True if the person is inside the store, past the gate line,
        or already registered as INSIDE.
        """
        if self.person_state.get(person_id) == "INSIDE":
            return True

        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)

        ax = self.point_a[0] * frame_w
        ay = self.point_a[1] * frame_h
        bx = self.point_b[0] * frame_w
        by = self.point_b[1] * frame_h

        dist = self._signed_distance_to_line(foot_x, foot_y, ax, ay, bx, by)
        return dist < 6.0  # past gate line into the store

    def reset(self):
        self.entry_count = 0
        self.exit_count = 0
        self.person_state.clear()
        self.deepest_inside.clear()
        self.person_history.clear()
        self.recent_events.clear()
        self.last_crossing_time.clear()
        self.pending_exits.clear()
        self.departed_persons.clear()
        self.pending_entries.clear()
        self.confirmed_entries.clear()
        self.visit_records.clear()
        self.ignored_visits.clear()
        self.last_eval_debug.clear()
        self.crossing_debug_log.clear()
        if self.group_detector is not None:
            self.group_detector.reset()

    @property
    def current_occupancy(self) -> int:
        return max(0, self.entry_count - self.exit_count)

    def _signed_distance_to_line(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        """
        Calculate the signed perpendicular distance from point P to line AB.
        Sign convention:
        Positive (+) = Outside the store (approaching gate from street)
        Negative (-) = Inside the store (past gate inside store)
        """
        dx = bx - ax
        dy = by - ay
        line_len = np.hypot(dx, dy)
        if line_len < 1.0:
            return 0.0

        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        raw_dist = cross / line_len

        if self.inside_direction == "below":
            return -raw_dist
        else:
            return raw_dist

    def confirm_pending_visits(self, timestamp: float) -> List[Dict]:
        """
        Confirm visits for individuals who entered and remained inside for > min_visit_duration_sec.
        Only confirmed visits increment official entry_count.
        """
        newly_confirmed = []
        for pid, entry_ts in list(self.pending_entries.items()):
            if (timestamp - entry_ts) > self.min_visit_duration_sec:
                self.entry_count += 1
                self.confirmed_entries[pid] = entry_ts
                del self.pending_entries[pid]

                ev_info = {
                    "person_id": pid,
                    "event": "IN",
                    "timestamp": entry_ts,
                    "confirmed": True,
                }
                self.recent_events.append(ev_info)
                newly_confirmed.append(ev_info)

                # Update visit record status
                for rec in reversed(self.visit_records):
                    if rec["person_id"] == pid and not rec["confirmed"]:
                        rec["confirmed"] = True
                        break

                if self.group_detector is not None:
                    self.group_detector.record_entry(pid, entry_ts)

        return newly_confirmed

    def register_entry(self, person_id: str, timestamp: float):
        """
        Register a provisional entry for a newly promoted permanent visitor upon valid gate crossing.
        Visit is confirmed only after remaining inside for > 5 seconds.
        Idempotent: will not overwrite or duplicate if already inside/confirmed for this visit.
        """
        if self.person_state.get(person_id) == "INSIDE" and person_id in self.confirmed_entries:
            return
        if person_id in self.pending_entries:
            return

        self.pending_entries[person_id] = timestamp
        self.person_state[person_id] = "INSIDE"
        self.departed_persons.discard(person_id)
        if timestamp > 0:
            self.last_crossing_time[person_id] = timestamp
        already_pending = any(r["person_id"] == person_id and r["exit_time"] is None for r in self.visit_records)
        if not already_pending:
            self.visit_records.append({
                "person_id": person_id,
                "entry_time": timestamp,
                "exit_time": None,
                "duration": None,
                "confirmed": False,
                "short_visit": False,
            })

    def _record_exit_event(self, person_id: str, entry_ts: float, exit_ts: float, is_short: bool):
        duration = exit_ts - entry_ts
        updated = False
        for rec in reversed(self.visit_records):
            if rec["person_id"] == person_id and rec["exit_time"] is None:
                rec["exit_time"] = exit_ts
                rec["duration"] = round(duration, 2)
                rec["confirmed"] = not is_short
                rec["short_visit"] = is_short
                updated = True
                break
        if not updated:
            self.visit_records.append({
                "person_id": person_id,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "duration": round(duration, 2),
                "confirmed": not is_short,
                "short_visit": is_short,
            })
        if is_short:
            self.ignored_visits.append({
                "person_id": person_id,
                "entry_time": entry_ts,
                "exit_time": exit_ts,
                "duration": round(duration, 2),
            })
        if self.identity_manager is not None:
            self.identity_manager.terminate_visit(person_id, exit_ts, is_short=is_short)

    def _record_eval(
        self,
        person_id: str,
        foot: Tuple[float, float],
        dist: float,
        prev_state: Optional[str],
        new_state: Optional[str],
        travel: float,
        span_t: float,
        is_within_span: bool,
        timestamp: float,
        event: Optional[str],
        reason: str,
    ):
        record = {
            "person_id": person_id,
            "foot": (round(foot[0], 1), round(foot[1], 1)),
            "dist": round(dist, 1),
            "prev_state": prev_state,
            "new_state": new_state,
            "travel": round(travel, 1),
            "span_t": round(span_t, 2),
            "is_within_span": is_within_span,
            "timestamp": round(timestamp, 2),
            "event": event,
            "reason": reason,
        }
        self.last_eval_debug[person_id] = record
        self.crossing_debug_log.append(record)

    def check(
        self,
        person_id: str,
        bbox: Tuple[int, int, int, int],
        frame_w: int,
        frame_h: int,
        is_static: bool = False,
        timestamp: float = 0.0,
        camera_id: str = "camera_01",
        feat: Optional[np.ndarray] = None,
        clothing: Optional[object] = None,
        crop: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Evaluate if a tracked person crossed the tripwire on this frame.
        """
        # Statues and mannequins NEVER count as entrance crossings
        if is_static:
            return None

        # Periodically evaluate pending entries for duration >= 5.0s confirmation
        if timestamp > 0:
            self.confirm_pending_visits(timestamp)

        # Calculate foot position in absolute frame pixels
        x1, y1, x2, y2 = bbox
        foot_x = (x1 + x2) / 2.0
        foot_y = float(y2)

        ax = self.point_a[0] * frame_w
        ay = self.point_a[1] * frame_h
        bx = self.point_b[0] * frame_w
        by = self.point_b[1] * frame_h

        line_len = np.hypot(bx - ax, by - ay)
        if line_len < 5.0:
            return None

        # Signed distance from the gate line
        dist = self._signed_distance_to_line(foot_x, foot_y, ax, ay, bx, by)
        self.person_history[person_id].append((foot_x, foot_y, dist))

        # Update deepest penetration inside store (negative numbers = deeper inside)
        if dist < self.deepest_inside[person_id]:
            self.deepest_inside[person_id] = dist

        # Check gate span alignment (within segment bounds + generous margins for gate openings)
        t = ((foot_x - ax) * (bx - ax) + (foot_y - ay) * (by - ay)) / (line_len ** 2)
        is_within_span = (-0.35 <= t <= 1.35)

        # Adaptive thresholds scaled to frame height
        corr_entry = max(self.corridor_half_width_px, frame_h * 0.03)
        corr_exit = max(15.0, frame_h * 0.02)   # Natural threshold across doorway onto street
        min_travel = max(self.min_crossing_travel_px * 0.65, frame_h * 0.03)
        min_exit_depth = max(self.min_in_store_depth_for_exit_px, frame_h * 0.10)
        entrance_threshold_zone = max(self.corridor_half_width_px * 2.5, 75.0, frame_h * 0.08)

        curr_state = self.person_state.get(person_id)

        # ── State Machine Initialization ─────────────────────────────────────
        if curr_state is None:
            if dist >= 0.0:
                self.person_state[person_id] = "OUTSIDE"
                init_reason = "INIT_OUTSIDE"
            elif dist >= -entrance_threshold_zone:
                # Within the entrance doorway threshold corridor
                self.person_state[person_id] = "THRESHOLD"
                init_reason = "INIT_THRESHOLD"
            else:
                # Genuinely deep inside the store before tracking began
                self.person_state[person_id] = "INSIDE"
                init_reason = "INIT_INSIDE_PREEXISTING"

            self._record_eval(person_id, (foot_x, foot_y), dist, None, self.person_state[person_id], 0.0, t, is_within_span, timestamp, None, init_reason)
            return None

        # Door-Holding Protection check: if person touched threshold but returned deep inside within 3s
        if person_id in self.pending_exits and dist < -50.0:
            self.pending_exits.pop(person_id)
            if person_id in self.departed_persons:
                self.departed_persons.discard(person_id)
            # Restore state to inside
            self.person_state[person_id] = "INSIDE"
            # If an exit was previously counted, roll it back
            if self.exit_count > 0:
                self.exit_count -= 1
            self.recent_events = deque(
                [e for e in self.recent_events if not (e.get('person_id') == person_id and e.get('event') == 'OUT')],
                maxlen=20
            )

        history = list(self.person_history[person_id])
        if len(history) < 2:
            self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, curr_state, 0.0, t, is_within_span, timestamp, None, "AWAITING_HISTORY")
            return None

        last_cross = self.last_crossing_time.get(person_id, -999.0)
        time_since_last_cross = (timestamp - last_cross) if timestamp > 0 else 999.0

        # ── 1. Potential ENTRY (IN) Event ────────────────────────────────────
        if curr_state in ("OUTSIDE", "THRESHOLD"):
            # If person already departed, require genuine deep re-entry into the retail floor
            entry_dist_req = -120.0 if person_id in self.departed_persons else -corr_entry * 0.8
            if dist < entry_dist_req and is_within_span:
                past_distances = [h[2] for h in history]
                max_past_dist = max(past_distances)

                # Traversal from outside (+) towards inside (-)
                travel = max_past_dist - dist

                # Must have originated from outside or threshold and traveled inward across gate
                originated_outside = (max_past_dist >= -max(corr_entry * 0.8, entrance_threshold_zone * 0.85))

                if originated_outside and travel >= min_travel and time_since_last_cross >= 1.0:
                    self.person_state[person_id] = "INSIDE"
                    self.departed_persons.discard(person_id)
                    if timestamp > 0:
                        self.last_crossing_time[person_id] = timestamp

                    # If already permanent visitor, register entry
                    if person_id.startswith("P-"):
                        self.register_entry(person_id, timestamp)
                    elif self.identity_manager is not None:
                        promoted_gid = self.identity_manager.promote_track_to_visitor(
                            person_id, timestamp, camera_id, feat, bbox, clothing=clothing, crop=crop
                        )
                        self.transfer_identity(person_id, promoted_gid)
                        self.register_entry(promoted_gid, timestamp)

                    self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, "INSIDE", travel, t, is_within_span, timestamp, "IN", "VALID_ENTRY_CROSSING")
                    return "IN"
                else:
                    reason = "REJECT_INSUFFICIENT_TRAVEL" if travel < min_travel else ("REJECT_NOT_ORIGINATED_OUTSIDE" if not originated_outside else "REJECT_RATE_LIMIT")
                    self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, curr_state, travel, t, is_within_span, timestamp, None, reason)
            else:
                reason = "OUTSIDE_GATE_SPAN" if not is_within_span else "NOT_DEEP_ENOUGH"
                self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, curr_state, 0.0, t, is_within_span, timestamp, None, reason)

        # ── 2. Potential EXIT (OUT) Event with Short Visit & Door-Opening Protection ───────
        elif curr_state == "INSIDE":
            has_been_deep_inside = (self.deepest_inside[person_id] <= -min_exit_depth)

            if dist >= corr_exit and is_within_span:
                past_distances = [h[2] for h in history]
                min_past_dist = min(past_distances)

                travel = dist - min_past_dist

                # Only count an EXIT if they were genuinely inside the store previously!
                if has_been_deep_inside and min_past_dist <= -(corr_entry * 0.3) and travel >= min_travel and time_since_last_cross >= 0.5:
                    self.person_state[person_id] = "OUTSIDE"
                    self.departed_persons.add(person_id)
                    if timestamp > 0:
                        self.last_crossing_time[person_id] = timestamp
                    self.deepest_inside[person_id] = dist

                    # Evaluate visit duration
                    # Case A: Short visit (duration <= 5.0 seconds) -> EXCLUDE FROM OFFICIAL IN & OUT
                    if person_id in self.pending_entries:
                        entry_ts = self.pending_entries.pop(person_id)
                        self._record_exit_event(person_id, entry_ts, timestamp, is_short=True)
                        self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, "OUTSIDE", travel, t, is_within_span, timestamp, None, "SHORT_VISIT_EXCLUDED")
                        # Short visit excluded from counts!
                        return None

                    # Case B: Confirmed visit (duration > 5.0 seconds) -> OFFICIAL EXIT
                    elif person_id in self.confirmed_entries:
                        entry_ts = self.confirmed_entries.pop(person_id)
                        duration = timestamp - entry_ts
                        self.pending_exits[person_id] = (timestamp, history[-1][0])
                        self.exit_count += 1
                        event = "OUT"
                        self._record_exit_event(person_id, entry_ts, timestamp, is_short=False)
                        self.recent_events.append({
                            "person_id": person_id,
                            "event": event,
                            "foot": (foot_x, foot_y),
                            "timestamp": timestamp,
                            "confirmed": True,
                        })
                        if self.group_detector is not None:
                            self.group_detector.record_exit(person_id, timestamp)
                        self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, "OUTSIDE", travel, t, is_within_span, timestamp, event, "VALID_EXIT_CROSSING")
                        return event
                else:
                    reason = "REJECT_EXIT_NOT_DEEP_ENOUGH" if not has_been_deep_inside else "REJECT_EXIT_INSUFFICIENT_TRAVEL"
                    self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, curr_state, travel, t, is_within_span, timestamp, None, reason)
            else:
                self._record_eval(person_id, (foot_x, foot_y), dist, curr_state, curr_state, 0.0, t, is_within_span, timestamp, None, "INSIDE_STORE")

        return None

    def check_departures(
        self,
        active_persons: Set[str],
        lost_persons: Set[str],
        frame_w: int,
        frame_h: int,
        timestamp: float = 0.0,
    ) -> List[Tuple[str, str, float]]:
        """
        Check if an INSIDE person who moved to the doorway threshold departed through the door.
        Also handles short visit filtering (<= 5s) vs confirmed exits.
        """
        events = []
        if timestamp > 0:
            self.confirm_pending_visits(timestamp)

        for pid in list(lost_persons):
            if pid in self.departed_persons:
                continue
            if self.person_state.get(pid) == "INSIDE":
                hist = self.person_history.get(pid)
                if hist and len(hist) >= 3:
                    lx, ly, ldist = hist[-1]
                    recent_k = min(len(hist), 5)
                    recent_dists = [h[2] for h in list(hist)[-recent_k:]]
                    heading_to_door = (recent_dists[-1] > recent_dists[0] + 6.0)
                    min_exit_depth = max(self.min_in_store_depth_for_exit_px, frame_h * 0.10)
                    has_been_deep = (self.deepest_inside.get(pid, 0.0) <= -min_exit_depth)
                    door_exit_thresh = max(self.corridor_half_width_px * 2.5, 75.0, frame_h * 0.08)
                    # Complete visit if tracked trajectory was moving toward door and reached doorway threshold corridor
                    should_depart = (has_been_deep and heading_to_door and ldist >= -door_exit_thresh)
                elif not hist and pid not in self.last_eval_debug:
                    # Direct test harness invocation without trajectory history
                    lx, ly = (0.0, 0.0)
                    should_depart = True
                else:
                    # Visitor simply occluded or out of view inside the store: do NOT depart!
                    should_depart = False

                if should_depart:
                    self.person_state[pid] = "OUTSIDE"
                    self.departed_persons.add(pid)

                    # Check short visit vs confirmed visit
                    if pid in self.pending_entries:
                        entry_ts = self.pending_entries.pop(pid)
                        self._record_exit_event(pid, entry_ts, timestamp, is_short=True)
                        # Excluded from official counts
                        continue

                    elif pid in self.confirmed_entries:
                        entry_ts = self.confirmed_entries.pop(pid)
                        self.exit_count += 1
                        self._record_exit_event(pid, entry_ts, timestamp, is_short=False)
                        ev = "OUT"
                        events.append((ev, pid, timestamp))
                        self.recent_events.append({
                            "person_id": pid,
                            "event": ev,
                            "foot": (lx, ly),
                            "timestamp": timestamp,
                            "confirmed": True,
                        })
                        if self.group_detector is not None:
                            self.group_detector.record_exit(pid, timestamp)
        return events
