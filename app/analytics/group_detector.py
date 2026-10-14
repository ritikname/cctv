import numpy as np
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple


class GroupDetector:
    """
    Exit-based group detection system.
    
    Assigns a shared group ID only when:
      1. Individually tracked people exit together (within exit_window_sec).
      2. Their entry timestamps are within 2 seconds of each other.
      3. The same number of individually tracked people who entered together exit together.
    
    Individual IDs (P-001, P-002, etc.) and individual visitor counts remain unchanged.
    """
    def __init__(
        self,
        entry_window_sec: float = 2.0,
        exit_window_sec: float = 2.0,
        history_frames: int = 10,
        max_distance: float = 150.0,
        static_speed: float = 5.0,
        direction_cosine: float = 0.5,
    ):
        self.entry_window_sec = entry_window_sec
        self.exit_window_sec = exit_window_sec
        self.history_frames = history_frames
        self.max_distance = max_distance
        self.static_speed = static_speed
        self.direction_cosine = direction_cosine

        self.entry_times: Dict[str, float] = {}   # person_id -> entry_timestamp
        self.exit_times: Dict[str, float] = {}    # person_id -> exit_timestamp
        self.recent_exits: List[Tuple[str, float]] = []  # list of (person_id, exit_timestamp)

        self.groups: Dict[str, int] = {}          # person_id -> group_id
        self.next_group_id: int = 1
        self.group_records: List[Dict] = []       # metadata for formed groups

        # Spatial history for visualization compatibility
        self.history = defaultdict(lambda: deque(maxlen=self.history_frames))

    def reset(self):
        """Reset all tracking and group records."""
        self.entry_times.clear()
        self.exit_times.clear()
        self.recent_exits.clear()
        self.groups.clear()
        self.next_group_id = 1
        self.group_records.clear()
        self.history.clear()

    def record_entry(self, person_id: str, timestamp: float):
        """Record an official entry timestamp for an individually tracked person."""
        self.entry_times[person_id] = timestamp
        # Clear previous visit group status so returning visitor can participate in a new group
        self.groups.pop(person_id, None)
        self.recent_exits = [(pid, ts) for pid, ts in self.recent_exits if pid != person_id]

    def record_exit(self, person_id: str, timestamp: float):
        """
        Record an official exit timestamp for an individually tracked person
        and evaluate exit-based group formation.
        """
        self.exit_times[person_id] = timestamp
        self.recent_exits.append((person_id, timestamp))
        self._evaluate_exit_groups(timestamp)

    def _evaluate_exit_groups(self, current_timestamp: float):
        """
        Evaluate recent exits to identify if an entire entry cohort exited together.
        """
        # Clean up very old unassigned exits (> 15 seconds ago)
        self.recent_exits = [
            (pid, ts) for pid, ts in self.recent_exits
            if (current_timestamp - ts) <= 15.0 or pid in self.groups
        ]

        # Consider candidates who do not yet have a group
        ungrouped_exits = [
            (pid, ts) for pid, ts in self.recent_exits
            if pid not in self.groups and pid in self.entry_times
        ]

        if len(ungrouped_exits) < 2:
            return

        # Cluster exiting individuals by exit timestamp
        # Two exits are in the same exit event if |t_out1 - t_out2| <= exit_window_sec
        clusters: List[List[Tuple[str, float]]] = []
        sorted_exits = sorted(ungrouped_exits, key=lambda x: x[1])

        for pid, t_exit in sorted_exits:
            matched_cluster = None
            for c in clusters:
                # Compare to cluster range
                c_times = [x[1] for x in c]
                if abs(t_exit - min(c_times)) <= self.exit_window_sec and abs(t_exit - max(c_times)) <= self.exit_window_sec:
                    matched_cluster = c
                    break
            if matched_cluster is not None:
                matched_cluster.append((pid, t_exit))
            else:
                clusters.append([(pid, t_exit)])

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            cluster_pids = [x[0] for x in cluster]
            # Verify entry timestamps: all members in cluster must have entry timestamps within entry_window_sec (2s)
            cluster_entries = [self.entry_times[pid] for pid in cluster_pids]
            entry_span = max(cluster_entries) - min(cluster_entries)
            if entry_span > self.entry_window_sec:
                # Entry timestamps not within 2 seconds -> reject grouping
                continue

            min_entry = min(cluster_entries)
            max_entry = max(cluster_entries)

            # Check full entry cohort:
            # Were there any other people who entered in this same entry window?
            # "Assign a shared group ID only when the same number of individually tracked people exit together"
            entry_cohort = [
                pid for pid, t_in in self.entry_times.items()
                if abs(t_in - min_entry) <= self.entry_window_sec and abs(t_in - max_entry) <= self.entry_window_sec
            ]

            if len(entry_cohort) != len(cluster_pids):
                # The number of people exiting together does not match the number of people who entered together!
                # (e.g. 3 entered together, but only 2 exited together)
                continue

            # Check that cluster_pids is an exact match for entry_cohort
            if set(cluster_pids) != set(entry_cohort):
                continue

            # Verified! Assign a shared group ID
            gid = self.next_group_id
            self.next_group_id += 1

            for pid in cluster_pids:
                self.groups[pid] = gid

            self.group_records.append({
                "group_id": gid,
                "members": list(cluster_pids),
                "entry_times": {pid: self.entry_times[pid] for pid in cluster_pids},
                "exit_times": {pid: self.exit_times[pid] for pid in cluster_pids},
                "entry_span_sec": round(entry_span, 3),
                "exit_span_sec": round(max(x[1] for x in cluster) - min(x[1] for x in cluster), 3),
            })

    def get_group(self, person_id: str) -> Optional[int]:
        """Return the group ID assigned to a person, or None."""
        return self.groups.get(person_id)

    def update(self, active_persons=None):
        """
        Maintains backward compatibility with processor loop.
        Returns the dictionary of confirmed group assignments (person_id -> group_id).
        """
        if active_persons:
            for p in active_persons:
                gid = p.get('global_id')
                bbox = p.get('bbox')
                if gid and bbox:
                    cx = (bbox[0] + bbox[2]) / 2.0
                    cy = float(bbox[3])
                    self.history[gid].append((cx, cy))

        return self.groups
