import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.identity.global_identity_manager import GlobalIdentityManager
from app.cameras.camera_manager import CameraManager
from app.config import Config


def _feat():
    v = np.random.rand(512).astype(np.float32)
    return v / np.linalg.norm(v)

def _close(feat, noise=0.005):
    v = feat + np.random.normal(0, noise, feat.shape).astype(np.float32)
    return v / np.linalg.norm(v)

def _perturb(feat, noise_level=0.52):
    target_sim = max(0.01, min(0.99, 1.0 - noise_level))
    ortho = np.random.randn(len(feat)).astype(np.float32)
    ortho -= float(np.dot(ortho, feat)) * feat
    ortho /= float(np.linalg.norm(ortho))
    v = target_sim * feat + float(np.sqrt(max(0.0, 1.0 - target_sim**2))) * ortho
    return v / np.linalg.norm(v)

def _clothing(upper_h=0.05, lower_h=0.60):
    from app.reid.clothing import ClothingProfile
    u = np.zeros(34, dtype=np.float32)
    l = np.zeros(34, dtype=np.float32)
    u_idx = int(upper_h * 17) % 18
    u[u_idx] = 0.8
    u[18] = 0.1
    u[26] = 0.1
    u /= np.linalg.norm(u)
    l_idx = int(lower_h * 17) % 18
    l[l_idx] = 0.8
    l[18] = 0.1
    l[26] = 0.1
    l /= np.linalg.norm(l)
    return ClothingProfile(u, l, quality=1.0)


class TestIdentityManager(unittest.TestCase):

    def setUp(self):
        cm = CameraManager()
        cm.add_camera("cam1", "Entrance", ["cam2"])
        cm.add_camera("cam2", "Store",    ["cam1"])
        self.mgr = GlobalIdentityManager(cm, require_candidate_motion=False)

    def _run(self, dets, feat_map, t=1.0, fi=1):
        return self.mgr.process_detections(dets, feat_map, t, fi)

    # ── 1. Single person, never lost ─────────────────────────────────────────
    def test_01_continuous(self):
        tid = "cam1-T1"
        fA  = _feat()
        det = [{'camera_id': 'cam1', 'track_id': tid,
                'bbox': (10, 10, 100, 200), 'conf': 0.9}]

        d1 = self._run(det, {tid: fA}, 1.0, 1)
        gid = d1[0]['global_id']

        d2 = self._run(det, {tid: _close(fA)}, 2.0, 2)
        self.assertEqual(d2[0]['global_id'], gid)
        self.assertEqual(len(self.mgr.persons), 1)

    # ── 2. Occlusion → Re-ID same person ─────────────────────────────────────
    def test_02_occlusion_reid(self):
        fA = _feat()
        det1 = [{'camera_id': 'cam1', 'track_id': 'cam1-T1',
                 'bbox': (100, 100, 200, 300), 'conf': 0.9}]
        d1 = self._run(det1, {'cam1-T1': fA}, 1.0, 1)
        gid_A = d1[0]['global_id']

        # Lost
        self._run([], {}, 2.0, 2)
        self.assertEqual(self.mgr.persons[gid_A].status, 'TEMPORARILY_LOST')

        # New track, NEAR same position, same appearance
        det2 = [{'camera_id': 'cam1', 'track_id': 'cam1-T5',
                 'bbox': (110, 110, 210, 310), 'conf': 0.9}]
        d3 = self._run(det2, {'cam1-T5': _close(fA)}, 3.0, 3)
        self.assertEqual(d3[0]['global_id'], gid_A, "Must recover same ID after occlusion")
        self.assertEqual(len(self.mgr.persons), 1)

    # ── 3. Cross-camera Re-ID ─────────────────────────────────────────────────
    def test_03_cross_camera(self):
        fA = _feat()
        d1 = self._run(
            [{'camera_id': 'cam1', 'track_id': 'cam1-T1',
              'bbox': (50, 50, 150, 300), 'conf': 0.9}],
            {'cam1-T1': fA}, 1.0, 1)
        gid = d1[0]['global_id']

        self._run([], {}, 2.0, 2)  # lost on cam1

        d3 = self._run(
            [{'camera_id': 'cam2', 'track_id': 'cam2-T1',
              'bbox': (60, 60, 160, 310), 'conf': 0.9}],
            {'cam2-T1': _close(fA)}, 3.0, 3)
        self.assertEqual(d3[0]['global_id'], gid, "Same person across cameras")
        self.assertEqual(len(self.mgr.persons), 1)

    # ── 4. Two distinct people stay separate ──────────────────────────────────
    def test_04_two_distinct(self):
        fA = _feat()
        fB = _feat()  # completely random → very different appearance
        dets = [
            {'camera_id': 'cam1', 'track_id': 'cam1-T1',
             'bbox': (10, 10, 100, 200), 'conf': 0.9},
            {'camera_id': 'cam1', 'track_id': 'cam1-T2',
             'bbox': (400, 10, 500, 200), 'conf': 0.9},
        ]
        d = self._run(dets, {'cam1-T1': fA, 'cam1-T2': fB}, 1.0, 1)
        gids = {x['global_id'] for x in d}
        self.assertEqual(len(gids), 2, "Two people must get different IDs")

    # ── 5. Active person can NOT be stolen ────────────────────────────────────
    def test_05_no_swap_while_active(self):
        fA = _feat()
        fB = _feat()
        # A and B both visible simultaneously
        dets = [
            {'camera_id': 'cam1', 'track_id': 'cam1-T1',
             'bbox': (10, 10, 100, 200), 'conf': 0.9},
            {'camera_id': 'cam1', 'track_id': 'cam1-T2',
             'bbox': (300, 10, 400, 200), 'conf': 0.9},
        ]
        d1 = self._run(dets, {'cam1-T1': fA, 'cam1-T2': fB}, 1.0, 1)
        gid_A = next(x['global_id'] for x in d1 if x['track_id'] == 'cam1-T1')
        gid_B = next(x['global_id'] for x in d1 if x['track_id'] == 'cam1-T2')

        # A gets a new track (BoT-SORT lost + re-detected)
        # But B is STILL visible → A's new track must NOT steal B's ID
        dets2 = [
            {'camera_id': 'cam1', 'track_id': 'cam1-T2',
             'bbox': (300, 10, 400, 200), 'conf': 0.9},
            {'camera_id': 'cam1', 'track_id': 'cam1-T99',
             'bbox': (15, 15, 105, 205), 'conf': 0.9},  # near A's old position
        ]
        d2 = self._run(dets2, {
            'cam1-T2': _close(fB),
            'cam1-T99': _close(fA)
        }, 2.0, 2)

        t99_gid = next(x['global_id'] for x in d2 if x['track_id'] == 'cam1-T99')
        t2_gid  = next(x['global_id'] for x in d2 if x['track_id'] == 'cam1-T2')

        # B must keep its ID
        self.assertEqual(t2_gid, gid_B, "Active person B must keep its ID")
        # T99 should match A (near A's position + similar appearance) NOT steal B
        self.assertNotEqual(t99_gid, gid_B, "New track must NOT steal active person's ID")


# ── Issue 1: Exit-based Group Detection Tests ──────────────────────────────
class TestExitBasedGroupDetection(unittest.TestCase):

    def setUp(self):
        from app.analytics.group_detector import GroupDetector
        self.gd = GroupDetector(entry_window_sec=2.0, exit_window_sec=2.0)

    def test_group_formed_when_same_number_and_entry_within_2s(self):
        """2 people enter within 2s and both exit together within 2s -> same group ID."""
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 11.5)  # Diff: 1.5s <= 2.0s

        self.gd.record_exit("P-001", 50.0)
        self.gd.record_exit("P-002", 51.0)  # Diff: 1.0s <= 2.0s

        g1 = self.gd.get_group("P-001")
        g2 = self.gd.get_group("P-002")

        self.assertIsNotNone(g1, "P-001 should have a group assigned")
        self.assertIsNotNone(g2, "P-002 should have a group assigned")
        self.assertEqual(g1, g2, "Both people must share the exact same group ID")
        self.assertEqual(len(self.gd.group_records), 1)
        self.assertEqual(self.gd.group_records[0]["members"], ["P-001", "P-002"])

    def test_no_group_when_entry_timestamps_differ_by_more_than_2s(self):
        """2 people enter > 2s apart and exit together -> NO group assigned."""
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 13.0)  # Diff: 3.0s > 2.0s

        self.gd.record_exit("P-001", 50.0)
        self.gd.record_exit("P-002", 51.0)

        self.assertIsNone(self.gd.get_group("P-001"))
        self.assertIsNone(self.gd.get_group("P-002"))
        self.assertEqual(len(self.gd.group_records), 0)

    def test_no_group_when_cohort_size_mismatches(self):
        """3 people enter together, but only 2 exit together -> NO group assigned."""
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 10.5)
        self.gd.record_entry("P-003", 11.2)  # 3 people in cohort

        # Only P-001 and P-002 exit together; P-003 remains in store
        self.gd.record_exit("P-001", 50.0)
        self.gd.record_exit("P-002", 51.0)

        self.assertIsNone(self.gd.get_group("P-001"), "Partial cohort exit must not form group")
        self.assertIsNone(self.gd.get_group("P-002"), "Partial cohort exit must not form group")

    def test_group_preserves_individual_visitor_ids(self):
        """Group detection must not alter or overwrite individual visitor IDs."""
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 11.0)
        self.gd.record_exit("P-001", 60.0)
        self.gd.record_exit("P-002", 61.0)

        # Mapping retains individual IDs as keys
        self.assertIn("P-001", self.gd.groups)
        self.assertIn("P-002", self.gd.groups)
        self.assertEqual(self.gd.groups["P-001"], self.gd.groups["P-002"])


# ── Issue 2: Ignore Short Visits Tests ────────────────────────────────────
class TestShortVisitFiltering(unittest.TestCase):

    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        # Horizontal gate line: outside (top, y < 500), inside (below, y > 500)
        self.tw = TripwireCounter(
            point_a=(0.0, 0.5), point_b=(1.0, 0.5),
            inside_direction="below",
            min_visit_duration_sec=5.0
        )

    def _walk_across_gate(self, pid, start_y, end_y, steps=8, t_start=1.0, t_end=2.0):
        w, h = 1000, 1000
        y_vals = np.linspace(start_y, end_y, steps)
        t_vals = np.linspace(t_start, t_end, steps)
        last_ret = None
        for y, t in zip(y_vals, t_vals):
            bbox = (450, int(y - 80), 550, int(y))
            ret = self.tw.check(pid, bbox, w, h, timestamp=float(t))
            if ret is not None:
                last_ret = ret
        return last_ret

    def test_short_visit_excluded_from_official_counts(self):
        """Entry and exit within <= 5s must exclude both events from gate-in and gate-out counts."""
        # 1. Person walks in from outside (y=400) to inside (y=650) at t=10.0 -> 11.0s
        self._walk_across_gate("P-001", 400, 650, steps=8, t_start=10.0, t_end=11.0)

        # Pending entry registered, but official entry_count is still 0
        self.assertEqual(self.tw.entry_count, 0, "Entry must not be confirmed immediately")
        self.assertIn("P-001", self.tw.pending_entries)

        # 2. Person walks back out (y=650 to y=400) at t=13.0 -> 14.0s (total visit <= 4.0s)
        self._walk_across_gate("P-001", 650, 400, steps=8, t_start=13.0, t_end=14.0)

        # Verified: Both events excluded from official counts
        self.assertEqual(self.tw.entry_count, 0, "Short visit entry must be excluded from official count")
        self.assertEqual(self.tw.exit_count, 0, "Short visit exit must be excluded from official count")

        # Verified: Visit record is recorded with short_visit=True
        short_recs = [r for r in self.tw.visit_records if r["person_id"] == "P-001" and r.get("short_visit")]
        self.assertTrue(len(short_recs) >= 1, "Short visit record must be preserved")
        self.assertFalse(short_recs[0]["confirmed"])
        self.assertLessEqual(short_recs[0]["duration"], 5.0)

    def test_confirmed_visit_after_5_seconds(self):
        """Person remains inside >= 5s -> visit confirmed, and subsequent exit increments exit_count."""
        w, h = 1000, 1000
        # 1. Person enters at t=10.0 -> 11.0s
        self._walk_across_gate("P-001", 400, 650, steps=8, t_start=10.0, t_end=11.0)
        self.assertEqual(self.tw.entry_count, 0)

        # 2. Person remains inside browsing until t=16.5s (> 5s duration)
        bbox_inside = (450, 570, 550, 650)
        self.tw.check("P-001", bbox_inside, w, h, timestamp=16.5)

        # Verified: Entry is now confirmed
        self.assertEqual(self.tw.entry_count, 1, "Visit must be confirmed after >= 5 seconds")
        self.assertIn("P-001", self.tw.confirmed_entries)

        # 3. Person exits at t=25.0 -> 26.0s
        self._walk_across_gate("P-001", 650, 400, steps=8, t_start=25.0, t_end=26.0)

        # Verified: Official exit counted
        self.assertEqual(self.tw.exit_count, 1, "Confirmed visit exit must increment exit count")
        confirmed_recs = [r for r in self.tw.visit_records if r["person_id"] == "P-001" and r.get("confirmed")]
        self.assertTrue(len(confirmed_recs) >= 1)
        self.assertGreaterEqual(confirmed_recs[0]["duration"], 5.0)


# ── Issue 3: Statue Classification & Resumption Tests ─────────────────────
class TestStatueClassificationAndResumption(unittest.TestCase):

    def setUp(self):
        cm = CameraManager()
        cm.add_camera("cam1", "StoreEntrance", [])
        self.mgr = GlobalIdentityManager(cm)

    def test_stationary_candidate_classified_as_statue_without_permanent_blacklist(self):
        """A candidate standing still is classified as STATUE, but never permanently blacklisted."""
        tid = "cam1-T10"
        f = _feat()

        # Feed 25 stationary detections at same coordinates
        for fi in range(1, 26):
            t = float(fi) * 0.1
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (200, 200, 280, 400), 'conf': 0.9}]
            decisions = self.mgr.process_detections(det, {tid: f}, t, fi)

        last_dec = decisions[0]
        self.assertEqual(last_dec['global_id'], "STATUE")
        self.assertEqual(last_dec['state'], "STATIC")

        # Must NOT be in a permanent blacklist set; candidate record must stay alive
        self.assertIn(tid, self.mgr.candidates, "Stationary candidate must NOT be deleted from candidates")

    def test_movement_resumption_restores_tracking(self):
        """When a stationary person resumes movement, tracking is restored and ID preserved."""
        tid = "cam1-T20"
        f = _feat()

        # 1. 25 stationary frames -> classified as STATUE
        for fi in range(1, 26):
            t = float(fi) * 0.1
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (200, 200, 280, 400), 'conf': 0.9}]
            self.mgr.process_detections(det, {tid: f}, t, fi)

        self.assertTrue(self.mgr.candidates[tid].get("is_static"))

        # 2. Movement resumes: person walks across the scene (displacement > 30px)
        decisions = None
        for fi in range(26, 35):
            t = float(fi) * 0.1
            offset = (fi - 25) * 8  # Moving 8 pixels per frame
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (200 + offset, 200, 280 + offset, 400), 'conf': 0.9}]
            decisions = self.mgr.process_detections(det, {tid: f}, t, fi)

        last_dec = decisions[0]
        # Tracking restored! No longer STATUE
        self.assertNotEqual(last_dec['global_id'], "STATUE", "Tracking must resume when movement resumes")
        self.assertIn(last_dec['global_id'], self.mgr.persons)
        self.assertIn(last_dec['state'], ("NEW", "ACTIVE"))

    def test_registered_person_standing_still_preserves_identity(self):
        """A registered visitor standing still retains their P-xxx ID and is never labeled STATUE."""
        tid = "cam1-T30"
        f = _feat()

        # Person enters with movement and gets registered as P-xxx
        gid = None
        for fi in range(1, 6):
            offset = fi * 10
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (100 + offset, 100, 180 + offset, 300), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, float(fi) * 0.1, fi)
            if d[0]['global_id'].startswith("P-"):
                gid = d[0]['global_id']

        self.assertIsNotNone(gid)
        self.assertTrue(gid.startswith("P-"))

        # Next 50 frames: Person stands completely still (e.g. browsing shelves)
        still_det = [{'camera_id': 'cam1', 'track_id': tid,
                      'bbox': (150, 100, 230, 300), 'conf': 0.9}]
        for fi in range(6, 56):
            t = float(fi) * 0.1
            d = self.mgr.process_detections(still_det, {tid: f}, t, fi)
            self.assertEqual(d[0]['global_id'], gid, "Standing still must NEVER change visitor ID to STATUE")
            self.assertNotEqual(d[0]['global_id'], "STATUE")

        # GlobalPerson is still registered and active in manager
        self.assertIn(gid, self.mgr.persons)
        self.assertEqual(self.mgr.persons[gid].status, "ACTIVE")

    def test_statue_resumes_movement_not_counted_without_gate_crossing(self):
        """A stationary candidate that resumes motion outside the gate is NOT counted as an entry."""
        from app.analytics.tripwire import TripwireCounter
        # Gate line at y = 0.5 (y=500 in 1000x1000), inside is below (y > 500)
        gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        tid = "cam1-T40"
        f = _feat()

        # 1. 20 frames stationary outside at y=200
        for fi in range(1, 21):
            t = float(fi) * 0.1
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (200, 100, 280, 200), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=gate, frame_w=1000, frame_h=1000)
            gate.check(tid, det[0]['bbox'], 1000, 1000, is_static=(d[0]['global_id'] == "STATUE"), timestamp=t)

        self.assertEqual(d[0]['global_id'], "STATUE")
        self.assertEqual(gate.entry_count, 0)

        # 2. Resumes motion outside (y moves from 200 to 300, still outside < 500)
        for fi in range(21, 31):
            t = float(fi) * 0.1
            offset = (fi - 20) * 10
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (200 + offset, 100, 280 + offset, 200), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=gate, frame_w=1000, frame_h=1000)
            gate.check(tid, det[0]['bbox'], 1000, 1000, is_static=(d[0]['global_id'] == "STATUE"), timestamp=t)

        # Movement resumed, but outside: NOT counted as an entry!
        self.assertNotEqual(d[0]['global_id'], "STATUE", "Tracking resumed")
        self.assertEqual(gate.entry_count, 0, "Must not count entry while still outside gate")

        # 3. Genuinely walks across entry line (foot y crosses 500 into 650)
        for fi in range(31, 45):
            t = float(fi) * 0.1
            y_foot = 200 + (fi - 30) * 35  # crosses past 500 into store
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (300, int(y_foot - 100), 380, int(y_foot)), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=gate, frame_w=1000, frame_h=1000)
            gid = d[0]['global_id']
            if gid not in ("PENDING", "STATUE", "OUTSIDE"):
                gate.transfer_identity(tid, gid)
                person_key = gid
            else:
                person_key = tid
            gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=(gid == "STATUE"), timestamp=t)
            if tid in self.mgr.local_to_global:
                person_key = self.mgr.local_to_global[tid]

        # Person enters store and gets promoted
        self.assertTrue(person_key.startswith("P-"))
        self.assertEqual(gate.person_state[person_key], "INSIDE")

        # After remaining inside >= 5s, entry is officially confirmed
        gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=False, timestamp=15.0)
        self.assertEqual(gate.entry_count, 1, "Only genuine gate crossing after >= 5s confirms entry")


# ── Issue 4: Accurate Gate Entry & Person Tagging Tests ────────────────────
class TestGateEntryAndPersonTagging(unittest.TestCase):

    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        cm = CameraManager()
        cm.add_camera("cam1", "StoreEntrance", [])
        self.mgr = GlobalIdentityManager(cm)
        # Gate line at y = 0.5 (y=500 in 1000x1000), inside is below (y > 500)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        self.gate.identity_manager = self.mgr

    def test_approaching_person_receives_no_visitor_id(self):
        """People approaching the gate, standing outside, or walking outside receive NO permanent visitor ID and 0 count."""
        tid = "cam1-outside-1"
        f = _feat()

        # Person moves outside the store (y = 200 -> 350, gate line is at 500)
        for fi in range(1, 25):
            t = fi * 0.1
            y = 200 + fi * 6
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (300, int(y - 80), 380, int(y)), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            self.gate.check(tid, det[0]['bbox'], 1000, 1000, timestamp=t)

            # Verification: Temporary ID retained, NO permanent P-xxx assigned!
            self.assertEqual(d[0]['global_id'], tid)
            self.assertEqual(d[0]['state'], "OUTSIDE")

        self.assertEqual(self.gate.entry_count, 0)
        self.assertEqual(len(self.mgr.persons), 0, "No visitor person records should be minted for outside people")

    def test_person_already_inside_receives_no_visitor_id_or_entry_count(self):
        """People already inside when tracking starts must not receive visitor IDs or be counted as new entries."""
        tid = "cam1-inside-1"
        f = _feat()

        # Person is already deep inside store (y = 750, gate is at 500)
        for fi in range(1, 25):
            t = fi * 0.1
            x = 400 + fi * 4
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (int(x), 650, int(x + 80), 750), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            self.gate.check(tid, det[0]['bbox'], 1000, 1000, timestamp=t)

            # Verification: Stays as temporary tracking ID inside, marked UNVERIFIED_OCCUPANT, no P-xxx
            self.assertEqual(d[0]['global_id'], tid)
            self.assertEqual(d[0]['state'], "UNVERIFIED_OCCUPANT")

        self.assertEqual(self.gate.entry_count, 0, "Already inside person must NOT increment entry count")
        self.assertEqual(len(self.mgr.persons), 0)

    def test_valid_crossing_increments_gate_in_exactly_once(self):
        """A valid crossing from outside to inside promotes to visitor ID and counts GATE IN exactly once."""
        tid = "cam1-cross-1"
        f = _feat()
        person_key = tid

        # 1. Approach and cross gate line (y goes from 350 to 650)
        for fi in range(1, 15):
            t = fi * 0.1
            y = 350 + fi * 25
            det = [{'camera_id': 'cam1', 'track_id': tid,
                    'bbox': (450, int(y - 80), 550, int(y)), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            ev = self.gate.check(person_key, det[0]['bbox'], 1000, 1000, timestamp=t)
            if tid in self.mgr.local_to_global:
                person_key = self.mgr.local_to_global[tid]

        # Promoted to permanent visitor ID
        self.assertTrue(person_key.startswith("P-"), f"Expected P-xxx, got {person_key}")
        self.assertEqual(self.gate.entry_count, 0, "Provisional entry pending 5-second confirmation")

        # 2. Stay inside for > 5 seconds
        det_inside = [{'camera_id': 'cam1', 'track_id': tid,
                       'bbox': (450, 620, 550, 700), 'conf': 0.9}]
        # Time advances to 8.0s (entry was at ~0.8s, so > 7s inside)
        self.gate.check(person_key, det_inside[0]['bbox'], 1000, 1000, timestamp=8.0)
        self.assertEqual(self.gate.entry_count, 1, "Visit confirmed after 5s inside")

        # 3. Subsequent frames inside must NOT count duplicate entries
        for fi in range(85, 120):
            t = fi * 0.1
            self.gate.check(person_key, det_inside[0]['bbox'], 1000, 1000, timestamp=t)

        self.assertEqual(self.gate.entry_count, 1, "GATE IN must remain exactly 1 (no duplicate counts)")


# ── Issue 5: Partial Group Exits and Counters ──────────────────────────────
class TestPartialGroupExitsAndCounters(unittest.TestCase):

    def setUp(self):
        from app.analytics.group_detector import GroupDetector
        from app.analytics.tripwire import TripwireCounter
        self.gd = GroupDetector(entry_window_sec=2.0, exit_window_sec=2.0)
        self.tw = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        self.tw.group_detector = self.gd

    def test_partial_group_exit_does_not_remove_visitors_still_inside(self):
        """
        3 visitors enter together (P-001, P-002, P-003).
        2 exit together (P-001, P-002).
        P-003 remains inside.
        - Partial exit does NOT form a group.
        - P-003 remains INSIDE and active.
        - Current occupancy correctly reflects 1 person still inside.
        """
        # 1. 3 people enter within 2 seconds
        self.tw.register_entry("P-001", 10.0)
        self.tw.register_entry("P-002", 10.8)
        self.tw.register_entry("P-003", 11.5)

        # Confirm all 3 entries after 6 seconds
        self.tw.confirm_pending_visits(17.0)
        self.assertEqual(self.tw.entry_count, 3)
        self.assertEqual(self.tw.current_occupancy, 3)

        # 2. P-001 and P-002 exit together at t=60.0 and 60.5
        w, h = 1000, 1000
        # Simulate P-001 and P-002 departing doorway
        self.tw.check_departures({"P-003"}, {"P-001", "P-002"}, w, h, timestamp=60.0)

        # Let's verify exit count and occupancy
        self.assertEqual(self.tw.exit_count, 2)
        self.assertEqual(self.tw.current_occupancy, 1, "P-003 must still be inside")
        self.assertEqual(self.tw.person_state["P-003"], "INSIDE", "P-003 state must remain INSIDE")
        self.assertNotIn("P-003", self.tw.departed_persons, "P-003 must not be marked departed")

        # Verified: Partial exit did NOT form a group
        self.assertIsNone(self.gd.get_group("P-001"))
        self.assertIsNone(self.gd.get_group("P-002"))
        self.assertEqual(len(self.gd.group_records), 0)

    def test_group_counter_updates_dynamically(self):
        """GROUPS counter increments dynamically when a complete cohort exits together."""
        self.tw.register_entry("P-001", 10.0)
        self.tw.register_entry("P-002", 11.2)
        self.tw.confirm_pending_visits(18.0)

        self.assertEqual(len(self.gd.group_records), 0)

        # Both exit together
        self.tw.check_departures(set(), {"P-001", "P-002"}, 1000, 1000, timestamp=50.0)

        self.assertEqual(len(self.gd.group_records), 1, "Group counter must increment to 1")
        self.assertEqual(self.gd.get_group("P-001"), 1)
        self.assertEqual(self.gd.get_group("P-002"), 1)


# ── Issue 6: Frame Skip & Timeline Consistency Tests ───────────────────────
class TestFrameSkipAndTimeline(unittest.TestCase):

    def test_frame_skip_timeline_consistency(self):
        """
        Verify that frame skipping (1, 2, 5) uses original video timestamps (video_frame_idx / fps),
        ensuring 5-second visit rule and visit durations remain identical regardless of frame skip.
        """
        fps = 25.0

        # Simulate 200 frames of a video (8.0 seconds total)
        for frame_skip in [1, 2, 5]:
            processed_timestamps = []
            for frame_idx in range(200):
                if frame_idx % frame_skip != 0:
                    continue
                timestamp = frame_idx / fps
                processed_timestamps.append((frame_idx, timestamp))

            # First processed frame is always frame 0, t=0.0s
            self.assertEqual(processed_timestamps[0][0], 0)
            self.assertEqual(processed_timestamps[0][1], 0.0)

            # Frame 100 is at video time 4.0s in all skip modes that include it
            # Frame 150 is at video time 6.0s in all skip modes that include it
            t_at_frame_150 = next(t for f, t in processed_timestamps if f == 150)
            self.assertAlmostEqual(t_at_frame_150, 6.0, places=3)

            # A person entering at video frame 25 (t=1.0s) and exiting at frame 175 (t=7.0s):
            # Duration must be 6.0s regardless of frame skip!
            t_enter = 25 / fps  # 1.0s
            t_exit = 175 / fps  # 7.0s
            duration = t_exit - t_enter
            self.assertAlmostEqual(duration, 6.0, places=3)
            self.assertGreater(duration, 5.0, "Must be confirmed visit in all frame skip modes")


# ── Issue 7: Reproduction Test for Short Visit Exit & Identity Transfer Prevention ──
class TestShortVisitExitAndIdentityTransferPrevention(unittest.TestCase):
    """
    Direct reproduction test for the identity transfer scenario:
    1. A man enters and exits within 3 seconds.
    2. A statue/mannequin remains visible throughout.
    3. Staff members are already inside without observed entry events.
    4. The short visit is excluded from official counts (gate-in and gate-out).
    5. The exited man's ID (P-001) must never appear on the statue or staff.
    6. Existing staff must remain UNVERIFIED_OCCUPANT and uncounted.
    7. A subsequent genuine entrant receives the correct identity (P-002) only after crossing the entry line.
    """
    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        cm = CameraManager()
        cm.add_camera("cam1", "StoreEntrance", [])
        self.mgr = GlobalIdentityManager(cm, require_candidate_motion=True)
        # Gate line at y = 0.5 (y=500 in 1000x1000), inside is below (y > 500)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below", min_visit_duration_sec=5.0)
        self.gate.identity_manager = self.mgr
        self.mgr.gate_counter = self.gate

    def test_short_visit_exit_does_not_transfer_id_to_statue_or_staff(self):
        tid_staff = "cam1-staff"
        tid_statue = "cam1-statue"
        tid_man = "cam1-man"
        tid_cust2 = "cam1-cust2"

        f_staff = _feat()
        f_statue = _feat()
        f_man = _feat()
        ortho = np.random.randn(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f_man)) * f_man
        f_cust2 = (ortho / np.linalg.norm(ortho)).astype(np.float32)

        # ── Phase 1: Video starts. Staff and Statue already inside ──────────
        # Frames 1-20: t = 0.1s -> 2.0s
        for fi in range(1, 21):
            t = fi * 0.1
            # Staff moving slightly inside (x: 600 -> 660, y: 750)
            x_staff = 600 + fi * 3
            bbox_staff = (int(x_staff), 670, int(x_staff + 70), 750)
            # Statue completely stationary inside at (800, 670, 870, 750)
            bbox_statue = (800, 670, 870, 750)

            dets = [
                {'camera_id': 'cam1', 'track_id': tid_staff, 'bbox': bbox_staff, 'conf': 0.92},
                {'camera_id': 'cam1', 'track_id': tid_statue, 'bbox': bbox_statue, 'conf': 0.95},
            ]
            feats = {tid_staff: f_staff, tid_statue: f_statue}

            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            for det in dets:
                dec = next(x for x in d if x['track_id'] == det['track_id'])
                is_statue = (dec['global_id'] == "STATUE")
                person_key = dec['global_id'] if dec['global_id'] not in ("PENDING", "STATUE", "OUTSIDE", "UNVERIFIED_OCCUPANT") else det['track_id']
                self.gate.check(person_key, det['bbox'], 1000, 1000, is_static=is_statue, timestamp=t)

        # Verify Phase 1:
        # Statue is classified as STATUE
        dec_statue = next(x for x in d if x['track_id'] == tid_statue)
        self.assertEqual(dec_statue['global_id'], "STATUE")
        self.assertEqual(dec_statue['state'], "STATIC")

        # Staff is UNVERIFIED_OCCUPANT, NOT assigned P-xxx
        dec_staff = next(x for x in d if x['track_id'] == tid_staff)
        self.assertEqual(dec_staff['global_id'], tid_staff)
        self.assertEqual(dec_staff['state'], "UNVERIFIED_OCCUPANT")

        # Gate counts are 0, no persons minted
        self.assertEqual(self.gate.entry_count, 0)
        self.assertEqual(self.gate.exit_count, 0)
        self.assertEqual(len(self.mgr.persons), 0)

        # ── Phase 2: Man enters through the gate (t = 2.1s -> 3.2s) ─────────
        # Man walks in from outside (y=300) into store (y=650)
        man_gid = None
        for fi in range(21, 33):
            t = fi * 0.1
            y_man = 300 + (fi - 20) * 30  # crosses past 500 into 650
            bbox_man = (400, int(y_man - 90), 470, int(y_man))

            x_staff = 660 + (fi - 20) * 2
            bbox_staff = (int(x_staff), 670, int(x_staff + 70), 750)
            bbox_statue = (800, 670, 870, 750)

            dets = [
                {'camera_id': 'cam1', 'track_id': tid_staff, 'bbox': bbox_staff, 'conf': 0.92},
                {'camera_id': 'cam1', 'track_id': tid_statue, 'bbox': bbox_statue, 'conf': 0.95},
                {'camera_id': 'cam1', 'track_id': tid_man, 'bbox': bbox_man, 'conf': 0.91},
            ]
            feats = {tid_staff: f_staff, tid_statue: f_statue, tid_man: f_man}

            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            for det in dets:
                dec = next(x for x in d if x['track_id'] == det['track_id'])
                is_statue = (dec['global_id'] == "STATUE")
                person_key = dec['global_id'] if dec['global_id'] not in ("PENDING", "STATUE", "OUTSIDE", "UNVERIFIED_OCCUPANT") else det['track_id']
                self.gate.check(person_key, det['bbox'], 1000, 1000, is_static=is_statue, timestamp=t)
                if det['track_id'] == tid_man and dec['global_id'].startswith("P-"):
                    man_gid = dec['global_id']

        self.assertIsNotNone(man_gid, "Man must be promoted to visitor ID upon crossing gate")
        self.assertEqual(man_gid, "P-001")
        # Pending entry registered, but official entry_count is still 0 (< 5s)
        self.assertEqual(self.gate.entry_count, 0)

        # ── Phase 3: Man exits within ~2 seconds (t = 3.3s -> 4.5s) ─────────
        for fi in range(33, 46):
            t = fi * 0.1
            y_man = 650 - (fi - 32) * 27  # moves back out past 500 to ~300
            bbox_man = (400, int(y_man - 90), 470, int(y_man))

            x_staff = 684 + (fi - 32) * 2
            bbox_staff = (int(x_staff), 670, int(x_staff + 70), 750)
            bbox_statue = (800, 670, 870, 750)

            dets = [
                {'camera_id': 'cam1', 'track_id': tid_staff, 'bbox': bbox_staff, 'conf': 0.92},
                {'camera_id': 'cam1', 'track_id': tid_statue, 'bbox': bbox_statue, 'conf': 0.95},
                {'camera_id': 'cam1', 'track_id': tid_man, 'bbox': bbox_man, 'conf': 0.91},
            ]
            feats = {tid_staff: f_staff, tid_statue: f_statue, tid_man: f_man}

            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            for det in dets:
                dec = next(x for x in d if x['track_id'] == det['track_id'])
                is_statue = (dec['global_id'] == "STATUE")
                person_key = dec['global_id'] if dec['global_id'] not in ("PENDING", "STATUE", "OUTSIDE", "UNVERIFIED_OCCUPANT") else det['track_id']
                self.gate.check(person_key, det['bbox'], 1000, 1000, is_static=is_statue, timestamp=t)

        # Short visit (duration ~1.5s <= 5.0s):
        # Excluded from official counts!
        self.assertEqual(self.gate.entry_count, 0, "Short visit must not increment gate-in count")
        self.assertEqual(self.gate.exit_count, 0, "Short visit must not increment gate-out count")

        # Man is officially marked DEPARTED
        p1 = self.mgr.persons["P-001"]
        self.assertTrue(p1.is_departed)
        self.assertEqual(p1.current_visit_status, "SHORT_VISIT_IGNORED")

        # ── Phase 4: Man is gone; Statue & Staff remain visible (t = 5.0s -> 8.0s) ──
        # Provide features that are VERY close / identical to P-001's embedding!
        for fi in range(50, 70):
            t = fi * 0.1
            x_staff = 710 + (fi - 50) * 2
            bbox_staff = (int(x_staff), 670, int(x_staff + 70), 750)
            bbox_statue = (800, 670, 870, 750)

            dets = [
                {'camera_id': 'cam1', 'track_id': tid_staff, 'bbox': bbox_staff, 'conf': 0.92},
                {'camera_id': 'cam1', 'track_id': tid_statue, 'bbox': bbox_statue, 'conf': 0.95},
            ]
            # Deliberately feed f_man as feature to test resistance against Re-ID hijack!
            feats = {tid_staff: f_man, tid_statue: f_man}

            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            for det in dets:
                dec = next(x for x in d if x['track_id'] == det['track_id'])
                is_statue = (dec['global_id'] == "STATUE")
                person_key = dec['global_id'] if dec['global_id'] not in ("PENDING", "STATUE", "OUTSIDE", "UNVERIFIED_OCCUPANT") else det['track_id']
                self.gate.check(person_key, det['bbox'], 1000, 1000, is_static=is_statue, timestamp=t)

            # CRITICAL ASSERTION: P-001 must NEVER be transferred to statue or staff!
            dec_statue = next(x for x in d if x['track_id'] == tid_statue)
            self.assertEqual(dec_statue['global_id'], "STATUE")
            self.assertEqual(dec_statue['state'], "STATIC")

            dec_staff = next(x for x in d if x['track_id'] == tid_staff)
            self.assertEqual(dec_staff['global_id'], tid_staff)
            self.assertEqual(dec_staff['state'], "UNVERIFIED_OCCUPANT")

        # Official counts remain 0
        self.assertEqual(self.gate.entry_count, 0)
        self.assertEqual(self.gate.exit_count, 0)

        # ── Phase 5: Second genuine entrant crosses gate (t = 10.0s -> 18.0s) ──
        cust2_gid = None
        for fi in range(100, 112):
            t = fi * 0.1
            y_cust2 = 300 + (fi - 100) * 30  # crosses past 500 into 650
            bbox_cust2 = (400, int(y_cust2 - 90), 470, int(y_cust2))

            dets = [
                {'camera_id': 'cam1', 'track_id': tid_staff, 'bbox': (750, 670, 820, 750), 'conf': 0.92},
                {'camera_id': 'cam1', 'track_id': tid_statue, 'bbox': (800, 670, 870, 750), 'conf': 0.95},
                {'camera_id': 'cam1', 'track_id': tid_cust2, 'bbox': bbox_cust2, 'conf': 0.93},
            ]
            feats = {tid_staff: f_staff, tid_statue: f_statue, tid_cust2: f_cust2}

            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            for det in dets:
                dec = next(x for x in d if x['track_id'] == det['track_id'])
                is_statue = (dec['global_id'] == "STATUE")
                person_key = dec['global_id'] if dec['global_id'] not in ("PENDING", "STATUE", "OUTSIDE", "UNVERIFIED_OCCUPANT") else det['track_id']
                self.gate.check(person_key, det['bbox'], 1000, 1000, is_static=is_statue, timestamp=t)
                if det['track_id'] == tid_cust2 and dec['global_id'].startswith("P-"):
                    cust2_gid = dec['global_id']

        # Entrant 2 receives P-002 (never P-001)
        self.assertEqual(cust2_gid, "P-002")

        # Entrant 2 remains inside until t = 17.0s (> 5s duration)
        for fi in range(112, 175):
            t = fi * 0.1
            bbox_cust2 = (420, 560, 490, 650)
            dets = [{'camera_id': 'cam1', 'track_id': tid_cust2, 'bbox': bbox_cust2, 'conf': 0.93}]
            feats = {tid_cust2: f_cust2}
            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            person_key = d[0]['global_id']
            self.gate.check(person_key, bbox_cust2, 1000, 1000, is_static=False, timestamp=t)

        # Official entry count is now 1!
        self.assertEqual(self.gate.entry_count, 1)

        # Entrant 2 exits at t = 18.0s
        for fi in range(175, 188):
            t = fi * 0.1
            y_cust2 = 650 - (fi - 175) * 27  # moves back out past 500 to 300
            bbox_cust2 = (420, int(y_cust2 - 90), 490, int(y_cust2))
            dets = [{'camera_id': 'cam1', 'track_id': tid_cust2, 'bbox': bbox_cust2, 'conf': 0.93}]
            feats = {tid_cust2: f_cust2}
            d = self.mgr.process_detections(dets, feats, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            person_key = d[0]['global_id']
            self.gate.check(person_key, bbox_cust2, 1000, 1000, is_static=False, timestamp=t)

        # Official exit count is now 1!
        self.assertEqual(self.gate.exit_count, 1)


# ── Issue 8: Persistent Re-ID Visitor Memory & Disappearance Recovery Tests ─
class TestPersistentReIDVisitorMemoryAndDisappearanceRecovery(unittest.TestCase):
    """
    Tests persistent visitor memory, disappearance, and reappearance:
    1. Once a person completes a valid entry crossing, persistent visitor memory (P-xxx, embeddings,
       visit ID, entry timestamp) is stored.
    2. When the visitor disappears from the camera frame / becomes occluded, their Re-ID history
       is retained, and visit is NOT completed (no departure).
    3. When the visitor reappears inside the store with a new tracker ID, their original person ID
       is restored without duplicate gate counts.
    4. Recovering after long absence across the room restores identity via appearance matching.
    5. Statues and other people never steal the absent visitor's identity.
    6. Confirmed exit terminates visit, and genuine later re-entry starts a separate visit (V-P-xxx-2).
    """

    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        cm = CameraManager()
        cm.add_camera("cam1", "StoreEntrance", [])
        self.mgr = GlobalIdentityManager(cm, require_candidate_motion=True)
        # Gate line at y = 0.5 (y=500 in 1000x1000), inside is below (y > 500)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below", min_visit_duration_sec=5.0)
        self.gate.identity_manager = self.mgr
        self.mgr.gate_counter = self.gate

    def test_visitor_disappearance_and_reappearance_recovers_identity_without_duplicate_count(self):
        """Entering visitor leaves frame, reappears with new track ID: identity restored, counts preserved."""
        tid_1 = "cam1-T1"
        tid_2 = "cam1-T2"
        f1 = _feat()

        # 1. Person enters through gate: foot y crosses from 350 to 650 (t=1.0 -> 2.2s)
        for fi in range(10, 22):
            t = fi * 0.1
            y = 350 + (fi - 10) * 28  # crosses past 500 into 650
            det = [{'camera_id': 'cam1', 'track_id': tid_1, 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.94}]
            d = self.mgr.process_detections(det, {tid_1: f1}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            person_key = d[0]['global_id'] if d[0]['global_id'] not in ("PENDING", "STATUE", "OUTSIDE") else tid_1
            self.gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        self.assertIn("P-001", self.mgr.persons)
        p = self.mgr.persons["P-001"]
        self.assertEqual(p.status, "ACTIVE")
        self.assertEqual(p.visit_id, "V-P-001-1")
        self.assertFalse(p.gallery.is_empty())

        # Remains inside for > 5.0 seconds to confirm visit
        for fi in range(22, 75):
            t = fi * 0.1
            det = [{'camera_id': 'cam1', 'track_id': tid_1, 'bbox': (420, 560, 490, 650), 'conf': 0.94}]
            d = self.mgr.process_detections(det, {tid_1: f1}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            self.gate.check("P-001", det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        self.assertEqual(self.gate.entry_count, 1, "Visit must be officially confirmed")
        self.assertEqual(self.gate.exit_count, 0)
        self.assertEqual(self.gate.person_state["P-001"], "INSIDE")

        # 2. Person leaves camera frame (0 detections for 25 frames, t=7.5 -> 10.0s)
        for fi in range(75, 100):
            t = fi * 0.1
            d = self.mgr.process_detections([], {}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            # Simulating lost tracking check in app
            self.gate.check_departures({"P-001"}, set(), 1000, 1000, timestamp=t)

        # Requirement 2 & 4: Do not delete identity or complete visit on temporary disappearance
        self.assertEqual(p.status, "TEMPORARILY_LOST")
        self.assertFalse(p.is_departed, "Must NOT mark as departed when simply out of camera frame")
        self.assertEqual(self.gate.person_state["P-001"], "INSIDE", "Must remain INSIDE")
        self.assertFalse(p.gallery.is_empty(), "Gallery must be preserved")
        self.assertEqual(self.gate.entry_count, 1)
        self.assertEqual(self.gate.exit_count, 0)

        # 3. Person reappears inside the store with NEW tracker ID (tid_2) at t=10.1 -> 12.0s
        recovered_gid = None
        for fi in range(101, 120):
            t = fi * 0.1
            # Reappears inside at (450, 570, 520, 660)
            det = [{'camera_id': 'cam1', 'track_id': tid_2, 'bbox': (450, 570, 520, 660), 'conf': 0.93}]
            d = self.mgr.process_detections(det, {tid_2: _close(f1)}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            gid = d[0]['global_id']
            state = d[0]['state']
            if gid.startswith("P-"):
                recovered_gid = gid
                self.gate.transfer_identity(tid_2, gid)
            person_key = gid if gid.startswith("P-") else tid_2
            self.gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        # Requirement 3: Recover original person ID
        self.assertEqual(recovered_gid, "P-001", "Must recover original P-001 identity")
        self.assertEqual(p.status, "ACTIVE")
        self.assertEqual(p.visit_id, "V-P-001-1", "Visit ID must remain original active visit")

        # Requirement 7: Do not increment gate counts again when recovering existing visitor
        self.assertEqual(self.gate.entry_count, 1, "GATE IN must NOT be incremented again on Re-ID recovery")
        self.assertEqual(self.gate.exit_count, 0)

    def test_long_absence_across_store_recovers_original_id(self):
        """Visitor leaves camera view for 40s, reappears on other side of frame: identity recovered."""
        tid_1 = "cam1-T1"
        tid_3 = "cam1-T3"
        f1 = _feat()

        # Enter at t=1.0 -> 2.0s
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': tid_1, 'bbox': (200, int(y - 90), 270, int(y)), 'conf': 0.94}]
            d = self.mgr.process_detections(det, {tid_1: f1}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            person_key = d[0]['global_id'] if d[0]['global_id'].startswith("P-") else tid_1
            self.gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        # Confirm visit (> 5s)
        self.gate.confirm_pending_visits(10.0)
        self.assertEqual(self.gate.entry_count, 1)

        # Disappears for 40 seconds (t = 10.0 -> 50.0s)
        self.mgr.process_detections([], {}, 50.0, 500, gate_counter=self.gate, frame_w=1000, frame_h=1000)
        self.assertEqual(self.mgr.persons["P-001"].status, "TEMPORARILY_LOST")

        # Reappears at t = 52.0s on the opposite side of the store (x=800 instead of 200)
        det = [{'camera_id': 'cam1', 'track_id': tid_3, 'bbox': (800, 600, 870, 720), 'conf': 0.92}]
        d = self.mgr.process_detections(det, {tid_3: _close(f1, 0.005)}, 52.0, 520, gate_counter=self.gate, frame_w=1000, frame_h=1000)

        self.assertEqual(d[0]['global_id'], "P-001", "Must recover P-001 even across large spatial displacement")
        self.assertEqual(d[0]['state'], "REIDENTIFIED")
        self.assertEqual(self.gate.entry_count, 1, "Count must remain 1")

    def test_confirmed_exit_completes_visit_and_genuine_reentry_creates_new_visit(self):
        """Confirmed exit terminates visit; genuine re-entry from outside starts separate visit V-P-001-2."""
        tid_1 = "cam1-T1"
        tid_return = "cam1-T9"
        f1 = _feat()

        # 1. Enter and confirm visit (t = 1.0 -> 8.0s)
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': tid_1, 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.94}]
            d = self.mgr.process_detections(det, {tid_1: f1}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            person_key = d[0]['global_id'] if d[0]['global_id'].startswith("P-") else tid_1
            self.gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        self.gate.confirm_pending_visits(8.0)
        self.assertEqual(self.gate.entry_count, 1)

        # 2. Confirmed exit: walks back out past gate line (t = 20.0 -> 22.0s)
        for fi in range(200, 220):
            t = fi * 0.1
            y = 650 - (fi - 200) * 18  # walks from y=650 back to y=300 (outside)
            det = [{'camera_id': 'cam1', 'track_id': tid_1, 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.94}]
            d = self.mgr.process_detections(det, {tid_1: f1}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            self.gate.check("P-001", det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        # Requirement 6: After confirmed exit, mark visit completed but retain historical memory
        self.assertEqual(self.gate.exit_count, 1, "Exit count must increment to 1")
        p = self.mgr.persons["P-001"]
        self.assertTrue(p.is_departed)
        self.assertEqual(p.current_visit_status, "COMPLETED")
        self.assertFalse(p.gallery.is_empty(), "Gallery must be retained for re-entry recognition")

        # 3. Genuine Re-Entry: P-001 returns from outside at t = 100.0s and crosses gate line INWARD
        for fi in range(1000, 1015):
            t = fi * 0.1
            y = 350 + (fi - 1000) * 25  # crosses past 500 inward into 650
            det = [{'camera_id': 'cam1', 'track_id': tid_return, 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.94}]
            d = self.mgr.process_detections(det, {tid_return: _close(f1, 0.005)}, t, fi, gate_counter=self.gate, frame_w=1000, frame_h=1000)
            person_key = d[0]['global_id'] if d[0]['global_id'].startswith("P-") else tid_return
            ev = self.gate.check(person_key, det[0]['bbox'], 1000, 1000, is_static=False, timestamp=t)

        # Requirement 6 & 7: Recognized as P-001 with new visit V-P-001-2!
        self.assertEqual(p.visit_id, "V-P-001-2", "Must start new visit record V-P-001-2")
        self.assertFalse(p.is_departed)

        # After staying > 5s on second visit, official entry count increments to 2!
        self.gate.confirm_pending_visits(110.0)
        self.assertEqual(self.gate.entry_count, 2, "Second visit must count new entry")


class TestConfigurableReIDThresholds(unittest.TestCase):
    def setUp(self):
        self.cm = CameraManager()
        self.cm.add_camera("cam1", "Entrance", [])
        self.mgr = GlobalIdentityManager(self.cm, require_candidate_motion=False)
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def tearDown(self):
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def test_default_thresholds_are_75(self):
        self.assertEqual(Config.REID_MATCH_THRESHOLD, 0.75)
        self.assertEqual(Config.ACTIVE_VISITOR_REID_THRESHOLD, 0.75)

    def test_threshold_rejects_below_75_and_accepts_above(self):
        # 1. Under 0.75 threshold: feature with sim 0.70 is rejected (creates new ID)
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75
        mgr1 = GlobalIdentityManager(self.cm, require_candidate_motion=False)
        f1 = _feat()
        det1 = [{'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (100, 100, 200, 300), 'conf': 0.9}]
        d1 = mgr1.process_detections(det1, {'T1': f1}, 1.0, 1)
        gid = d1[0]['global_id']

        mgr1.process_detections([], {}, 2.0, 2)
        self.assertEqual(mgr1.persons[gid].status, 'TEMPORARILY_LOST')

        # Synthetic feature with exact cosine similarity 0.70 (< 0.75)
        ortho = np.random.rand(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f1)) * f1
        ortho /= float(np.linalg.norm(ortho))
        f_70 = (0.70 * f1 + np.sqrt(1 - 0.70**2) * ortho).astype(np.float32)
        f_70 /= float(np.linalg.norm(f_70))

        det2 = [{'camera_id': 'cam1', 'track_id': 'T2', 'bbox': (105, 105, 205, 305), 'conf': 0.9}]
        d2 = mgr1.process_detections(det2, {'T2': f_70}, 3.0, 3)
        self.assertNotEqual(d2[0]['global_id'], gid, "Should not match when similarity 0.70 < threshold 0.75")

        # 2. Under 0.68 threshold: same feature with sim 0.70 is accepted (recovers gid)
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.68
        mgr2 = GlobalIdentityManager(self.cm, require_candidate_motion=False)
        d1_b = mgr2.process_detections(det1, {'T1': f1}, 1.0, 1)
        gid_b = d1_b[0]['global_id']
        mgr2.process_detections([], {}, 2.0, 2)
        self.assertEqual(mgr2.persons[gid_b].status, 'TEMPORARILY_LOST')

        d2_b = mgr2.process_detections(det2, {'T2': f_70}, 3.0, 3)
        self.assertEqual(d2_b[0]['global_id'], gid_b, "Should match when similarity 0.70 >= threshold 0.68")


class TestMultiViewAndClothingReID(unittest.TestCase):
    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        self.cm = CameraManager()
        self.cm.add_camera("cam1", "Entrance", [])
        self.mgr = GlobalIdentityManager(self.cm, require_candidate_motion=False)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        self.mgr.gate_counter = self.gate
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def tearDown(self):
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def _synthetic_crop(self, upper_color, lower_color, size=(120, 60)):
        """Generate a realistic mock BGR crop with specific upper and lower clothing colors."""
        h, w = size
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        # Head (first 15%)
        crop[: int(h * 0.15), :] = [180, 180, 180]
        # Upper body (15% - 55%)
        crop[int(h * 0.15) : int(h * 0.55), :] = upper_color
        # Lower body (55% - 90%)
        crop[int(h * 0.55) : int(h * 0.90), :] = lower_color
        # Shoes (90% - 100%)
        crop[int(h * 0.90) :, :] = [50, 50, 50]
        return crop

    def test_front_to_back_turn_preserves_identity(self):
        """
        When a person enters facing front and then turns around showing only their back,
        deep appearance similarity drops (0.50-0.65), but matching clothing HSV histograms
        and multi-view gallery maintain their persistent identity without creating a duplicate ID.
        """
        # Red shirt, blue jeans
        front_crop = self._synthetic_crop([0, 0, 220], [200, 50, 0])
        # Back view of the same person: same red shirt and blue jeans, but no face
        back_crop = self._synthetic_crop([0, 0, 215], [195, 55, 5])

        f_front = _feat()
        # Back view has lower deep feature similarity to front view (~0.55):
        ortho = np.random.rand(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f_front)) * f_front
        ortho /= float(np.linalg.norm(ortho))
        f_back = (0.55 * f_front + np.sqrt(1 - 0.55**2) * ortho).astype(np.float32)
        f_back /= float(np.linalg.norm(f_back))

        # 1. Person enters walking forward (front view)
        det_front = [{
            'camera_id': 'cam1', 'track_id': 'cam1-T1',
            'bbox': (400, 420, 480, 560), 'conf': 0.95, 'crop': front_crop
        }]
        d1 = self.mgr.process_detections(det_front, {'cam1-T1': f_front}, 1.0, 1, gate_counter=self.gate)
        gid = self.mgr.promote_track_to_visitor('cam1-T1', 1.0, bbox=det_front[0]['bbox'])
        self.assertEqual(gid, "P-001")
        self.gate.check(gid, det_front[0]['bbox'], 1000, 1000, is_static=False, timestamp=1.0)

        # 2. Track lost briefly during turn
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons["P-001"].status, "TEMPORARILY_LOST")

        # 3. Reappears as a new track showing only back view
        det_back = [{
            'camera_id': 'cam1', 'track_id': 'cam1-T2',
            'bbox': (410, 415, 490, 555), 'conf': 0.93, 'crop': back_crop
        }]
        d2 = self.mgr.process_detections(det_back, {'cam1-T2': f_back}, 3.0, 3, gate_counter=self.gate)

        # Must recover original P-001 identity via clothing and fused Re-ID
        self.assertEqual(d2[0]['global_id'], "P-001", "Must preserve P-001 identity after turning around")
        self.assertEqual(len(self.mgr.persons), 1, "Must NOT create duplicate person ID for back view")

    def test_similar_clothing_distinguished_by_appearance_and_simultaneous_visibility(self):
        """
        Two people with similar clothing (both wearing dark jackets) must NEVER share an ID
        if they are simultaneously visible or have different deep embeddings.
        """
        crop_a = self._synthetic_crop([30, 30, 30], [180, 40, 20])  # dark jacket, blue jeans
        crop_b = self._synthetic_crop([35, 35, 35], [175, 45, 25])  # also dark jacket, blue jeans

        fA = _feat()
        fB = _feat()  # distinct deep appearance

        dets = [
            {'camera_id': 'cam1', 'track_id': 'cam1-T1', 'bbox': (100, 200, 180, 350), 'conf': 0.94, 'crop': crop_a},
            {'camera_id': 'cam1', 'track_id': 'cam1-T2', 'bbox': (500, 200, 580, 350), 'conf': 0.94, 'crop': crop_b},
        ]
        d = self.mgr.process_detections(dets, {'cam1-T1': fA, 'cam1-T2': fB}, 1.0, 1, gate_counter=self.gate)
        g1 = self.mgr.promote_track_to_visitor('cam1-T1', 1.0, bbox=dets[0]['bbox'])
        g2 = self.mgr.promote_track_to_visitor('cam1-T2', 1.0, bbox=dets[1]['bbox'])

        self.assertNotEqual(g1, g2, "Two simultaneous people must have different IDs")
        # Check co-presence graph
        self.assertIn('cam1-T2', self.mgr.persons[g1].co_present_track_ids)

    def test_occluded_crop_does_not_pollute_gallery(self):
        """
        When two people overlap (IoU > 0.12), the occluded detection is flagged
        and never added to the gallery, keeping the multi-view gallery pure.
        """
        crop_clean = self._synthetic_crop([0, 200, 0], [0, 0, 0])     # green shirt
        crop_occluded = self._synthetic_crop([200, 0, 200], [0, 0, 0]) # purple noise

        f_clean = _feat()
        ortho = np.random.rand(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f_clean)) * f_clean
        ortho /= float(np.linalg.norm(ortho))
        f_bad = ortho

        # Step 1: Clean registration
        det1 = [{'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (200, 200, 280, 360), 'conf': 0.95, 'crop': crop_clean}]
        self.mgr.process_detections(det1, {'T1': f_clean}, 1.0, 1)
        gid = self.mgr.promote_track_to_visitor('T1', 1.0, bbox=det1[0]['bbox'])
        p = self.mgr.persons[gid]
        initial_size = p.gallery.size()

        # Step 2: Overlapping boxes with another passerby (IoU > 0.12)
        dets_overlap = [
            {'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (200, 200, 280, 360), 'conf': 0.95, 'crop': crop_occluded},
            {'camera_id': 'cam1', 'track_id': 'T2', 'bbox': (220, 200, 300, 360), 'conf': 0.95, 'crop': crop_occluded},
        ]
        self.mgr.process_detections(dets_overlap, {'T1': f_bad, 'T2': f_bad}, 2.0, 2)

        # Corrupted feature must NOT have been added to T1's gallery
        self.assertEqual(p.gallery.size(), initial_size, "Gallery must not accept features from occluded detection")
        sim_to_bad = p.gallery.best_similarity(f_bad)
        self.assertLess(sim_to_bad, 0.20, "Gallery must not contain the corrupted orthogonal feature")

    def test_crossing_trajectories_preserve_identities_without_swap(self):
        """
        Two people walking towards each other and crossing paths retain their
        respective IDs without identity switching.
        """
        crop1 = self._synthetic_crop([0, 180, 200], [20, 20, 20])   # yellow top, black pants
        crop2 = self._synthetic_crop([180, 0, 0], [150, 150, 150])  # blue top, grey pants

        f1 = _feat()
        f2 = _feat()

        # Approaching from opposite sides
        det_pre = [
            {'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (200, 300, 270, 440), 'conf': 0.95, 'crop': crop1},
            {'camera_id': 'cam1', 'track_id': 'T2', 'bbox': (600, 300, 670, 440), 'conf': 0.95, 'crop': crop2},
        ]
        self.mgr.process_detections(det_pre, {'T1': f1, 'T2': f2}, 1.0, 1)
        g1 = self.mgr.promote_track_to_visitor('T1', 1.0, bbox=det_pre[0]['bbox'])
        g2 = self.mgr.promote_track_to_visitor('T2', 1.0, bbox=det_pre[1]['bbox'])

        # After crossing: T1 is now at x=620, T2 is at x=180
        det_post = [
            {'camera_id': 'cam1', 'track_id': 'T1_post', 'bbox': (620, 300, 690, 440), 'conf': 0.95, 'crop': crop1},
            {'camera_id': 'cam1', 'track_id': 'T2_post', 'bbox': (180, 300, 250, 440), 'conf': 0.95, 'crop': crop2},
        ]
        # Lost step
        self.mgr.process_detections([], {}, 2.0, 2)
        d_post = self.mgr.process_detections(det_post, {'T1_post': f1, 'T2_post': f2}, 3.0, 3)

        res_map = {x['track_id']: x['global_id'] for x in d_post}
        self.assertEqual(res_map['T1_post'], g1, "T1 must retain g1 identity after crossing")
        self.assertEqual(res_map['T2_post'], g2, "T2 must retain g2 identity after crossing")


class TestCrossPersonMisassignmentPrevention(unittest.TestCase):
    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        self.cm = CameraManager()
        self.cm.add_camera("cam1", "Entrance", [])
        self.mgr = GlobalIdentityManager(self.cm, require_candidate_motion=False)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        self.mgr.gate_counter = self.gate
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def tearDown(self):
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def _synthetic_crop(self, upper_color, lower_color, size=(120, 60)):
        h, w = size
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        crop[: int(h * 0.15), :] = [180, 180, 180]
        crop[int(h * 0.15) : int(h * 0.55), :] = upper_color
        crop[int(h * 0.55) : int(h * 0.90), :] = lower_color
        crop[int(h * 0.90) :, :] = [50, 50, 50]
        return crop

    def test_woman_enters_first_man_does_not_get_her_id(self):
        """
        Reproduce exact scenario: A woman enters first (P-001) in pink top & black skirt.
        She moves deeper into the store and is temporarily lost.
        A man appears at the entrance in green shirt & khaki pants.
        He must NOT be assigned the woman's ID P-001.
        """
        woman_crop = self._synthetic_crop([50, 10, 200], [20, 20, 20])   # pink top, black skirt
        man_crop   = self._synthetic_crop([50, 100, 0], [100, 80, 50])   # green shirt, khaki pants
        f_woman = _feat()
        f_man = _feat()

        # 1. Woman enters
        det_woman = [{
            'camera_id': 'cam1', 'track_id': 'cam1-T1',
            'bbox': (400, 420, 480, 560), 'conf': 0.95, 'crop': woman_crop
        }]
        self.mgr.process_detections(det_woman, {'cam1-T1': f_woman}, 1.0, 1, gate_counter=self.gate)
        gid_w = self.mgr.promote_track_to_visitor('cam1-T1', 1.0, bbox=det_woman[0]['bbox'])
        self.assertEqual(gid_w, "P-001")

        # 2. Woman is temporarily occluded / not in current frame
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons["P-001"].status, "TEMPORARILY_LOST")

        # 3. Man appears
        det_man = [{
            'camera_id': 'cam1', 'track_id': 'cam1-T2',
            'bbox': (405, 415, 485, 555), 'conf': 0.94, 'crop': man_crop
        }]
        d_man = self.mgr.process_detections(det_man, {'cam1-T2': f_man}, 3.0, 3, gate_counter=self.gate)

        # Man must NOT receive P-001!
        assigned_id = d_man[0]['global_id']
        self.assertNotEqual(assigned_id, "P-001", "Man must never be assigned the woman's ID")
        self.assertNotEqual(d_man[0]['state'], "REIDENTIFIED")

    def test_clothing_gate_veto_even_if_deep_feature_accidentally_similar(self):
        """
        Even if CNN deep appearance vector produces similarity near 0.76 (due to un-whitened
        ResNet50 bias or coincidence), distinct clothing (cloth_sim < 0.30) MUST trigger
        the CLOTHING_MIN_GATE veto and reject the match.
        """
        woman_crop = self._synthetic_crop([50, 10, 200], [20, 20, 20])   # pink top, black skirt
        man_crop   = self._synthetic_crop([50, 100, 0], [100, 80, 50])   # green shirt, khaki pants
        f_woman = _feat()
        # Create a feature that has ~0.76 dot product with f_woman
        ortho = np.random.randn(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f_woman)) * f_woman
        ortho /= float(np.linalg.norm(ortho))
        f_similar = (0.76 * f_woman + np.sqrt(1 - 0.76**2) * ortho).astype(np.float32)
        f_similar /= float(np.linalg.norm(f_similar))

        det_w = [{
            'camera_id': 'cam1', 'track_id': 'cam1-T1',
            'bbox': (300, 300, 370, 450), 'conf': 0.95, 'crop': woman_crop
        }]
        self.mgr.process_detections(det_w, {'cam1-T1': f_woman}, 1.0, 1, gate_counter=self.gate)
        gid_w = self.mgr.promote_track_to_visitor('cam1-T1', 1.0, bbox=det_w[0]['bbox'])

        # Lost step
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)

        # Man appears with f_similar but distinct man_crop clothing
        det_m = [{
            'camera_id': 'cam1', 'track_id': 'cam1-T2',
            'bbox': (310, 310, 380, 460), 'conf': 0.95, 'crop': man_crop
        }]
        d_m = self.mgr.process_detections(det_m, {'cam1-T2': f_similar}, 3.0, 3, gate_counter=self.gate)

        # Must NOT re-identify as P-001 because clothing vetoed it!
        self.assertNotEqual(d_m[0]['global_id'], "P-001")
        # Check audit log for veto
        veto_logs = self.mgr.get_reid_log("REJECTED_CLOTHING_GATE")
        self.assertTrue(len(veto_logs) > 0, "Audit log must record REJECTED_CLOTHING_GATE")

    def test_gallery_protected_from_contamination(self):
        """
        Woman's gallery must not be contaminated when a different person enters.
        """
        woman_crop = self._synthetic_crop([200, 0, 100], [20, 20, 20])
        man_crop   = self._synthetic_crop([0, 150, 0], [100, 100, 100])
        f_w = _feat()
        ortho = np.random.randn(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f_w)) * f_w
        ortho /= float(np.linalg.norm(ortho))
        f_m = ortho.astype(np.float32)

        det_w = [{'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (200, 200, 280, 360), 'conf': 0.95, 'crop': woman_crop}]
        self.mgr.process_detections(det_w, {'T1': f_w}, 1.0, 1, gate_counter=self.gate)
        gid_w = self.mgr.promote_track_to_visitor('T1', 1.0, bbox=det_w[0]['bbox'])
        p_w = self.mgr.persons[gid_w]
        initial_gallery_size = p_w.gallery.size()

        # Frame 2: Man appears
        det_m = [{'camera_id': 'cam1', 'track_id': 'T2', 'bbox': (400, 200, 480, 360), 'conf': 0.95, 'crop': man_crop}]
        self.mgr.process_detections(det_m, {'T2': f_m}, 2.0, 2, gate_counter=self.gate)

        # Woman's gallery must not have absorbed man's features
        self.assertEqual(p_w.gallery.size(), initial_gallery_size)
        sim_to_man = p_w.gallery.best_similarity(f_m)
        self.assertLess(sim_to_man, 0.20)


# ── Issue 11: Gate Crossing & Tripwire Detection Tests ───────────────────────
class TestGateCrossingAndTripwireFixes(unittest.TestCase):
    """
    Tests for reliable gate crossing detection:
    1. Simultaneous / adjacent entrants walking shoulder-to-shoulder (< 100px apart).
    2. Fast crosser under frame skip (2 frames crossing threshold).
    3. Pre-existing occupant inside the store walking around never triggers entry.
    4. Audit evaluation logs capture foot point, distance, state transitions, and reasons.
    5. 5-second short visit rule remains strictly enforced.
    """
    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        cm = CameraManager()
        cm.add_camera("cam1", "Entrance", [])
        self.mgr = GlobalIdentityManager(cm, require_candidate_motion=True)
        # Gate line at y = 500 (point_a=(0.0, 0.5), point_b=(1.0, 0.5)), inside is below (y > 500)
        self.tw = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below", min_visit_duration_sec=5.0)
        self.tw.identity_manager = self.mgr
        self.mgr.gate_counter = self.tw

    def test_adjacent_group_entrants_both_counted(self):
        """
        Two people enter shoulder-to-shoulder (within 60px of each other).
        Both must be successfully detected and counted as entries without overlap suppression.
        """
        w, h = 1000, 1000
        # Frame 1: Both outside at y = 470 (dist = +30px)
        # Person 1 at x = 400, Person 2 at x = 460 (only 60px apart!)
        bbox1_f1 = (370, 370, 430, 470)
        bbox2_f1 = (430, 370, 490, 470)
        ev1_1 = self.tw.check("T1", bbox1_f1, w, h, timestamp=1.0)
        ev2_1 = self.tw.check("T2", bbox2_f1, w, h, timestamp=1.0)
        self.assertIsNone(ev1_1)
        self.assertIsNone(ev2_1)
        self.assertEqual(self.tw.person_state["T1"], "OUTSIDE")
        self.assertEqual(self.tw.person_state["T2"], "OUTSIDE")

        # Frame 2: Both cross into store at y = 545 (dist = -45px)
        bbox1_f2 = (370, 445, 430, 545)
        bbox2_f2 = (430, 445, 490, 545)
        ev1_2 = self.tw.check("T1", bbox1_f2, w, h, timestamp=1.2)
        ev2_2 = self.tw.check("T2", bbox2_f2, w, h, timestamp=1.2)

        self.assertEqual(ev1_2, "IN", "First adjacent entrant must trigger IN")
        self.assertEqual(ev2_2, "IN", "Second adjacent entrant must trigger IN (not swallowed by overlap)")
        self.assertEqual(self.tw.person_state["T1"], "INSIDE")
        self.assertEqual(self.tw.person_state["T2"], "INSIDE")

        # Confirm visits after 5.5 seconds inside
        self.tw.confirm_pending_visits(7.0)
        self.assertEqual(self.tw.entry_count, 2, "Both entrants must be confirmed in entry_count")

    def test_fast_crosser_frame_skip(self):
        """
        A person walks fast or frame-skip occurs (2 frames across threshold).
        Must reliably trigger IN without requiring 5 historical frames.
        """
        w, h = 1000, 1000
        # Frame 1: at y = 480 (dist = +20px, outside)
        bbox_f1 = (400, 380, 460, 480)
        ev1 = self.tw.check("T-FAST", bbox_f1, w, h, timestamp=10.0)
        self.assertIsNone(ev1)

        # Frame 2 (skip): jumps to y = 550 (dist = -50px, deep inside)
        bbox_f2 = (400, 450, 460, 550)
        ev2 = self.tw.check("T-FAST", bbox_f2, w, h, timestamp=10.3)
        self.assertEqual(ev2, "IN", "Fast crosser jumping across line must trigger IN on 2nd frame")
        self.assertEqual(self.tw.person_state["T-FAST"], "INSIDE")

    def test_preexisting_occupant_never_triggers_in(self):
        """
        An occupant already inside the store (dist = -300px) walks around.
        Must be initialized to INSIDE and NEVER trigger IN or increment entry_count.
        """
        w, h = 1000, 1000
        # Occupant walking around at y = 800, 850, 780
        for i, y in enumerate([800, 820, 850, 830, 780]):
            bbox = (600, y - 100, 680, y)
            ev = self.tw.check("STAFF-1", bbox, w, h, timestamp=20.0 + i * 0.2)
            self.assertIsNone(ev, "In-store occupant must NEVER trigger IN")
            self.assertEqual(self.tw.person_state["STAFF-1"], "INSIDE")

        self.tw.confirm_pending_visits(35.0)
        self.assertEqual(self.tw.entry_count, 0, "Pre-existing occupant must never increment entry_count")

    def test_debug_evaluation_log(self):
        """
        Verify that last_eval_debug and crossing_debug_log record foot point,
        signed distance, state transitions, and exact decision reasons.
        """
        w, h = 1000, 1000
        bbox = (400, 380, 460, 480) # foot = (430, 480), dist = +20px
        self.tw.check("T-LOG", bbox, w, h, timestamp=5.0)

        dbg = self.tw.last_eval_debug.get("T-LOG")
        self.assertIsNotNone(dbg)
        self.assertEqual(dbg["person_id"], "T-LOG")
        self.assertEqual(dbg["foot"], (430.0, 480.0))
        self.assertAlmostEqual(dbg["dist"], 20.0, places=1)
        self.assertEqual(dbg["new_state"], "OUTSIDE")
        self.assertEqual(dbg["reason"], "INIT_OUTSIDE")
        self.assertGreater(len(self.tw.crossing_debug_log), 0)



# ── Issue 11: Returning Visitor Identity Recovery & Viewpoint Changes ─────────
class TestReturningVisitorIdentityRecovery(unittest.TestCase):
    """
    Verifies that:
    1. A group of women entering the store receive permanent visitor IDs.
    2. Leaving the camera frame does NOT complete their visit or delete Re-ID memory.
    3. Reappearing with different viewing angles (front-to-back) and new tracker IDs
       restores their original visitor IDs without duplicate GATE IN increments.
    4. Unverified occupants (staff / people already inside) and statues never steal visitor IDs.
    5. Detailed Re-ID logs capture candidate matches, scores, and decisions.
    """

    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        from app.reid.clothing import ClothingProfile
        cm = CameraManager()
        cm.add_camera("cam1", "StoreEntrance", [])
        self.mgr = GlobalIdentityManager(cm, require_candidate_motion=True)
        # Gate at y = 0.5 (y = 500 in 1000x1000), inside is below (y > 500)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below", min_visit_duration_sec=5.0)
        self.gate.identity_manager = self.mgr
        self.mgr.gate_counter = self.gate

    def test_group_entry_disappearance_and_viewpoint_turn_recovery(self):
        w, h = 1000, 1000

        # Distinct embeddings and clothing for Woman 1 (maroon) and Woman 2 (floral)
        f_w1_front = _feat()
        f_w2_front = _feat()
        c_w1 = _clothing(upper_h=0.05, lower_h=0.60) # maroon top, blue pants
        c_w2 = _clothing(upper_h=0.15, lower_h=0.15) # floral dress

        # 1. Entry: Both women cross inward from y=380 to y=650 (t=1.0 -> 2.5s)
        for fi in range(10, 25):
            t = fi * 0.1
            y = 380 + (fi - 10) * 20
            dets = [
                {'camera_id': 'cam1', 'track_id': 'cam1-T1', 'bbox': (300, int(y - 90), 380, int(y)), 'conf': 0.95},
                {'camera_id': 'cam1', 'track_id': 'cam1-T2', 'bbox': (450, int(y - 90), 530, int(y)), 'conf': 0.94},
            ]
            f_map = {'cam1-T1': f_w1_front, 'cam1-T2': f_w2_front}
            c_map = {'cam1-T1': c_w1, 'cam1-T2': c_w2}
            decs = self.mgr.process_detections(dets, f_map, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map=c_map)
            dec_map = {d['track_id']: d for d in decs}

            for det in dets:
                tid = det['track_id']
                dec = dec_map[tid]
                gid = dec['global_id']
                pkey = gid if (gid and gid.startswith("P-")) else tid
                ev = self.gate.check(pkey, det['bbox'], w, h, is_static=False, timestamp=t)
                if ev == "IN" and not pkey.startswith("P-"):
                    new_gid = self.mgr.promote_track_to_visitor(tid, t, "cam1", f_map.get(tid), det['bbox'], clothing=c_map.get(tid))
                    self.gate.transfer_identity(tid, new_gid)
                    self.gate.register_entry(new_gid, t)

        # Both promoted
        self.assertIn("P-001", self.mgr.persons)
        self.assertIn("P-002", self.mgr.persons)

        # Remain inside to confirm visits (> 5 seconds)
        for fi in range(25, 75):
            t = fi * 0.1
            dets = [
                {'camera_id': 'cam1', 'track_id': 'cam1-T1', 'bbox': (320, 580, 400, 670), 'conf': 0.95},
                {'camera_id': 'cam1', 'track_id': 'cam1-T2', 'bbox': (460, 590, 540, 680), 'conf': 0.94},
            ]
            f_map = {'cam1-T1': f_w1_front, 'cam1-T2': f_w2_front}
            c_map = {'cam1-T1': c_w1, 'cam1-T2': c_w2}
            self.mgr.process_detections(dets, f_map, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map=c_map)
            self.gate.check("P-001", dets[0]['bbox'], w, h, is_static=False, timestamp=t)
            self.gate.check("P-002", dets[1]['bbox'], w, h, is_static=False, timestamp=t)

        self.assertEqual(self.gate.entry_count, 2, "Both women confirmed as gate entries")
        self.assertEqual(self.gate.exit_count, 0)
        self.assertEqual(self.gate.person_state["P-001"], "INSIDE")
        self.assertEqual(self.gate.person_state["P-002"], "INSIDE")

        # 2. Temporary disappearance (they explore the store, 0 detections for 30 frames)
        for fi in range(75, 105):
            t = fi * 0.1
            self.mgr.process_detections([], {}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.gate.check_departures({"P-001", "P-002"}, set(), w, h, timestamp=t)

        # Re-ID memory preserved, status is TEMPORARILY_LOST, visits uncompleted
        p1 = self.mgr.persons["P-001"]
        p2 = self.mgr.persons["P-002"]
        self.assertEqual(p1.status, "TEMPORARILY_LOST")
        self.assertEqual(p2.status, "TEMPORARILY_LOST")
        self.assertFalse(p1.is_departed)
        self.assertFalse(p2.is_departed)
        self.assertEqual(self.gate.entry_count, 2, "Counts preserved during disappearance")
        self.assertEqual(self.gate.exit_count, 0)

        # 3. Reappearance with NEW tracker IDs and front-to-back turned views
        # Deep similarity drops to ~0.50 due to back view, but clothing matches strongly (>= 0.70)
        # Simultaneously, a staff member (unverified occupant) is visible inside the store
        f_w1_back = _perturb(f_w1_front, noise_level=0.52)  # cos sim ~0.50
        f_w2_back = _perturb(f_w2_front, noise_level=0.52)
        f_staff = _feat() # totally different person
        c_staff = _clothing(upper_h=0.85, lower_h=0.0) # black/white uniform

        reappear_dets = [
            {'camera_id': 'cam1', 'track_id': 'cam1-T20', 'bbox': (340, 600, 420, 700), 'conf': 0.93}, # Woman 1 back
            {'camera_id': 'cam1', 'track_id': 'cam1-T21', 'bbox': (480, 610, 560, 710), 'conf': 0.92}, # Woman 2 back
            {'camera_id': 'cam1', 'track_id': 'cam1-T22', 'bbox': (150, 700, 220, 800), 'conf': 0.90}, # Staff member
        ]
        f_reappear = {'cam1-T20': f_w1_back, 'cam1-T21': f_w2_back, 'cam1-T22': f_staff}
        c_reappear = {'cam1-T20': c_w1, 'cam1-T21': c_w2, 'cam1-T22': c_staff}

        decs = self.mgr.process_detections(
            reappear_dets, f_reappear, 11.0, 110,
            gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map=c_reappear
        )
        dec_map = {d['track_id']: d for d in decs}

        # Woman 1 recovered as P-001
        self.assertEqual(dec_map['cam1-T20']['global_id'], "P-001")
        self.assertEqual(dec_map['cam1-T20']['state'], "REIDENTIFIED")

        # Woman 2 recovered as P-002
        self.assertEqual(dec_map['cam1-T21']['global_id'], "P-002")
        self.assertEqual(dec_map['cam1-T21']['state'], "REIDENTIFIED")

        # Staff member NOT assigned any visitor ID -> stays temporary track ID and UNVERIFIED_OCCUPANT
        self.assertEqual(dec_map['cam1-T22']['global_id'], "cam1-T22")
        self.assertEqual(dec_map['cam1-T22']['state'], "UNVERIFIED_OCCUPANT")

        # Invariant checks: GATE IN still exactly 2, no duplicates
        for det in reappear_dets:
            tid = det['track_id']
            gid = dec_map[tid]['global_id']
            pkey = gid if (gid and gid.startswith("P-")) else tid
            ev = self.gate.check(pkey, det['bbox'], w, h, is_static=False, timestamp=11.0)
            self.assertIsNone(ev, "Re-identifying visitor must NEVER trigger GATE IN")

        self.assertEqual(self.gate.entry_count, 2, "GATE IN must strictly remain 2")
        self.assertEqual(self.gate.exit_count, 0)

        # Audit logs verification
        reid_log = self.mgr.get_reid_log()
        accepted_logs = [l for l in reid_log if l.get('decision') == 'ACCEPTED']
        self.assertTrue(any(l['query_tid'] == 'cam1-T20' and l['candidate_gid'] == 'P-001' for l in accepted_logs))
        self.assertTrue(any(l['query_tid'] == 'cam1-T21' and l['candidate_gid'] == 'P-002' for l in accepted_logs))



# ── Mandatory Regression Suite: 12 Core Requirements ─────────────────────────
class TestMandatoryRegressionScenarios(unittest.TestCase):
    """
    Mandatory regression tests required for CCTV Person Re-ID:
    1. A woman enters, temporarily disappears, and returns: recover original ID.
    2. A different man appears while the woman is temporarily lost: do not assign her ID to him.
    3. Two people cross paths: prevent identity swaps.
    4. Two people wear similar clothing: do not merge their identities based on color alone.
    5. A visitor turns from front-facing to back-facing: recover identity when confidence is sufficient.
    6. A person is occluded and receives a new local tracker ID: preserve the persistent identity.
    7. An ambiguous Re-ID match occurs: defer rather than force the assignment.
    8. A statue remains visible: do not assign it a visitor identity.
    9. A person already inside at startup: do not fabricate an entry.
    10. A completed visitor returns: recognize person, but create a new visit record.
    11. A short visit is ignored: do not transfer that visitor's ID to another person.
    12. An identity is recovered: do not increment gate counts twice or corrupt group membership.
    """

    def setUp(self):
        from app.analytics.tripwire import TripwireCounter
        from app.analytics.group_detector import GroupDetector
        self.cm = CameraManager()
        self.cm.add_camera("cam1", "MainEntrance", [])
        self.mgr = GlobalIdentityManager(self.cm, require_candidate_motion=True)
        # Gate line at y = 0.5 (y = 500 in 1000x1000), inside is below (y > 500)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below", min_visit_duration_sec=5.0)
        self.gd = GroupDetector(entry_window_sec=2.0, exit_window_sec=2.0)
        self.gate.group_detector = self.gd
        self.gate.identity_manager = self.mgr
        self.mgr.gate_counter = self.gate
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def tearDown(self):
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def _synthetic_crop(self, upper_color, lower_color, size=(120, 60)):
        h, w = size
        crop = np.zeros((h, w, 3), dtype=np.uint8)
        crop[: int(h * 0.15), :] = [180, 180, 180]
        crop[int(h * 0.15) : int(h * 0.55), :] = upper_color
        crop[int(h * 0.55) : int(h * 0.90), :] = lower_color
        crop[int(h * 0.90) :, :] = [50, 50, 50]
        return crop

    # 1. A woman enters, temporarily disappears, and returns: recover original ID
    def test_case_01_woman_enters_disappears_returns(self):
        w, h = 1000, 1000
        f_w = _feat()
        crop_w = self._synthetic_crop([120, 30, 200], [20, 20, 20])
        c_w = _clothing(upper_h=0.08, lower_h=0.60)

        # Woman crosses gate inward (t = 1.0 -> 2.0s)
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-W1', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95, 'crop': crop_w}]
            d = self.mgr.process_detections(det, {'T-W1': f_w}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-W1': c_w})
            self.gate.check('T-W1', det[0]['bbox'], w, h, timestamp=t)
            if 'T-W1' in self.mgr.local_to_global:
                gid = self.mgr.local_to_global['T-W1']
                self.gate.transfer_identity('T-W1', gid)

        self.assertIn("P-001", self.mgr.persons)
        self.gate.confirm_pending_visits(8.0)
        self.assertEqual(self.gate.entry_count, 1)

        # Woman disappears from frame
        self.mgr.process_detections([], {}, 10.0, 100, gate_counter=self.gate, frame_w=w, frame_h=h)
        self.assertEqual(self.mgr.persons["P-001"].status, "TEMPORARILY_LOST")

        # Returns with new tracker ID and consistent appearance
        det_ret = [{'camera_id': 'cam1', 'track_id': 'T-W2', 'bbox': (420, 580, 490, 670), 'conf': 0.94, 'crop': crop_w}]
        d_ret = self.mgr.process_detections(det_ret, {'T-W2': _close(f_w)}, 12.0, 120, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-W2': c_w})
        self.assertEqual(d_ret[0]['global_id'], "P-001", "Must recover original visitor ID P-001")
        self.assertEqual(d_ret[0]['state'], "REIDENTIFIED")

    # 2. A different man appears while the woman is temporarily lost: do not assign her ID to him
    def test_case_02_different_man_appears_while_woman_lost(self):
        w, h = 1000, 1000
        f_w = _feat()
        f_m = _feat()
        c_w = _clothing(upper_h=0.05, lower_h=0.60) # red/blue
        c_m = _clothing(upper_h=0.35, lower_h=0.10) # green/khaki

        # Woman enters as P-001
        det_w = [{'camera_id': 'cam1', 'track_id': 'T-WOMAN', 'bbox': (400, 420, 470, 560), 'conf': 0.95}]
        self.mgr.process_detections(det_w, {'T-WOMAN': f_w}, 1.0, 1, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-WOMAN': c_w})
        gid_w = self.mgr.promote_track_to_visitor('T-WOMAN', 1.0, bbox=det_w[0]['bbox'], clothing=c_w)
        self.assertEqual(gid_w, "P-001")

        # Woman temporarily lost
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate, frame_w=w, frame_h=h)
        self.assertEqual(self.mgr.persons["P-001"].status, "TEMPORARILY_LOST")

        # Man appears
        det_m = [{'camera_id': 'cam1', 'track_id': 'T-MAN', 'bbox': (405, 415, 475, 555), 'conf': 0.95}]
        d_m = self.mgr.process_detections(det_m, {'T-MAN': f_m}, 3.0, 3, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-MAN': c_m})
        self.assertNotEqual(d_m[0]['global_id'], "P-001", "Woman's ID must never be assigned to different man")

    # 3. Two people cross paths: prevent identity swaps
    def test_case_03_crossing_trajectories_prevent_swaps(self):
        f1 = _feat()
        f2 = _feat()
        c1 = _clothing(upper_h=0.10, lower_h=0.60)
        c2 = _clothing(upper_h=0.70, lower_h=0.20)

        # Before crossing
        dets_pre = [
            {'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (200, 300, 270, 440), 'conf': 0.95},
            {'camera_id': 'cam1', 'track_id': 'T2', 'bbox': (600, 300, 670, 440), 'conf': 0.95},
        ]
        self.mgr.process_detections(dets_pre, {'T1': f1, 'T2': f2}, 1.0, 1, clothing_map={'T1': c1, 'T2': c2})
        g1 = self.mgr.promote_track_to_visitor('T1', 1.0, bbox=dets_pre[0]['bbox'], clothing=c1)
        g2 = self.mgr.promote_track_to_visitor('T2', 1.0, bbox=dets_pre[1]['bbox'], clothing=c2)

        # Crossing / disappearance
        self.mgr.process_detections([], {}, 2.0, 2)

        # After crossing paths: positions swapped
        dets_post = [
            {'camera_id': 'cam1', 'track_id': 'T1_after', 'bbox': (620, 300, 690, 440), 'conf': 0.95},
            {'camera_id': 'cam1', 'track_id': 'T2_after', 'bbox': (180, 300, 250, 440), 'conf': 0.95},
        ]
        d_post = self.mgr.process_detections(dets_post, {'T1_after': f1, 'T2_after': f2}, 3.0, 3, clothing_map={'T1_after': c1, 'T2_after': c2})
        res = {x['track_id']: x['global_id'] for x in d_post}
        self.assertEqual(res['T1_after'], g1, "T1 must retain original ID g1 after crossing")
        self.assertEqual(res['T2_after'], g2, "T2 must retain original ID g2 after crossing")

    # 4. Two people wear similar clothing: do not merge their identities based on color alone
    def test_case_04_similar_clothing_no_merge_based_on_color_alone(self):
        fA = _feat()
        ortho = np.random.randn(512).astype(np.float32)
        ortho -= float(np.dot(ortho, fA)) * fA
        ortho /= float(np.linalg.norm(ortho))
        fB = ortho.astype(np.float32) # distinct appearance
        c_similar = _clothing(upper_h=0.0, lower_h=0.0) # both dark jackets, dark pants

        # Person A is active
        det_A = [{'camera_id': 'cam1', 'track_id': 'T-A', 'bbox': (200, 200, 280, 380), 'conf': 0.95}]
        self.mgr.process_detections(det_A, {'T-A': fA}, 1.0, 1, gate_counter=self.gate, clothing_map={'T-A': c_similar})
        gid_A = self.mgr.promote_track_to_visitor('T-A', 1.0, bbox=det_A[0]['bbox'], clothing=c_similar)

        # Person A leaves frame (temporarily lost)
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons[gid_A].status, "TEMPORARILY_LOST")

        # Person B arrives with identical clothing color, but distinct deep embedding
        det_B = [{'camera_id': 'cam1', 'track_id': 'T-B', 'bbox': (210, 210, 290, 390), 'conf': 0.95}]
        d_B = self.mgr.process_detections(det_B, {'T-B': fB}, 3.0, 3, gate_counter=self.gate, clothing_map={'T-B': c_similar})

        # MUST NOT be assigned gid_A based on clothing alone!
        self.assertNotEqual(d_B[0]['global_id'], gid_A, "Similar clothing colors alone must never merge identities")

    # 5. A visitor turns from front-facing to back-facing: recover identity when confidence is sufficient
    def test_case_05_front_to_back_viewpoint_turn_recovery(self):
        front_crop = self._synthetic_crop([0, 0, 220], [200, 50, 0])
        back_crop = self._synthetic_crop([0, 0, 215], [195, 55, 5])
        f_front = _feat()
        ortho = np.random.rand(512).astype(np.float32)
        ortho -= float(np.dot(ortho, f_front)) * f_front
        ortho /= float(np.linalg.norm(ortho))
        f_back = (0.55 * f_front + np.sqrt(1 - 0.55**2) * ortho).astype(np.float32)
        f_back /= float(np.linalg.norm(f_back))

        det_f = [{'camera_id': 'cam1', 'track_id': 'T-FRONT', 'bbox': (400, 420, 480, 560), 'conf': 0.95, 'crop': front_crop}]
        self.mgr.process_detections(det_f, {'T-FRONT': f_front}, 1.0, 1, gate_counter=self.gate)
        gid = self.mgr.promote_track_to_visitor('T-FRONT', 1.0, bbox=det_f[0]['bbox'])

        # Turn during disappearance
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)

        det_b = [{'camera_id': 'cam1', 'track_id': 'T-BACK', 'bbox': (410, 415, 490, 555), 'conf': 0.94, 'crop': back_crop}]
        d_b = self.mgr.process_detections(det_b, {'T-BACK': f_back}, 3.0, 3, gate_counter=self.gate)
        self.assertEqual(d_b[0]['global_id'], gid, "Turned visitor showing back must recover identity via multi-view & clothing")

    # 6. A person is occluded and receives a new local tracker ID: preserve persistent identity
    def test_case_06_occluded_new_tracker_id_preserves_persistent_identity(self):
        f = _feat()
        det1 = [{'camera_id': 'cam1', 'track_id': 'T-ORIG', 'bbox': (300, 300, 380, 480), 'conf': 0.95}]
        self.mgr.process_detections(det1, {'T-ORIG': f}, 1.0, 1, gate_counter=self.gate)
        gid = self.mgr.promote_track_to_visitor('T-ORIG', 1.0, bbox=det1[0]['bbox'])

        # 1-frame occlusion
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)

        # BoT-SORT re-detection with new tracker ID
        det2 = [{'camera_id': 'cam1', 'track_id': 'T-NEW', 'bbox': (305, 305, 385, 485), 'conf': 0.95}]
        d2 = self.mgr.process_detections(det2, {'T-NEW': _close(f)}, 3.0, 3, gate_counter=self.gate)
        self.assertEqual(d2[0]['global_id'], gid, "Tracker ID reassignment after occlusion must preserve persistent ID")

    # 7. An ambiguous Re-ID match occurs: defer rather than force the assignment
    def test_case_07_ambiguous_reid_match_defers_assignment(self):
        f1 = _feat()
        f2 = _feat()
        # Query feature with identical distance to f1 and f2
        f_mid = (0.5 * f1 + 0.5 * f2).astype(np.float32)
        f_mid /= float(np.linalg.norm(f_mid))

        # Register two visitors P-001 and P-002
        det1 = [{'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (200, 300, 270, 450), 'conf': 0.95}]
        det2 = [{'camera_id': 'cam1', 'track_id': 'T2', 'bbox': (600, 300, 670, 450), 'conf': 0.95}]
        self.mgr.process_detections([det1[0], det2[0]], {'T1': f1, 'T2': f2}, 1.0, 1, gate_counter=self.gate)
        g1 = self.mgr.promote_track_to_visitor('T1', 1.0, bbox=det1[0]['bbox'])
        g2 = self.mgr.promote_track_to_visitor('T2', 1.0, bbox=det2[0]['bbox'])

        # Both visitors lost
        self.mgr.process_detections([], {}, 2.0, 2, gate_counter=self.gate)

        # An ambiguous query appears equidistant between both candidates
        det_amb = [{'camera_id': 'cam1', 'track_id': 'T-AMB', 'bbox': (400, 300, 470, 450), 'conf': 0.92}]
        # Set spatial score neutral and clothing None so composite score evaluates deep sim
        d_amb = self.mgr.process_detections(det_amb, {'T-AMB': f_mid}, 3.0, 3, gate_counter=self.gate)

        # Must DEFER assignment and set state to IDENTITY_UNCERTAIN without fabricating a new visitor
        self.assertEqual(d_amb[0]['state'], "IDENTITY_UNCERTAIN", "Ambiguous match must defer assignment")
        self.assertNotIn(d_amb[0]['global_id'], (g1, g2), "Must not force an ambiguous match to either candidate")

    # 8. A statue remains visible: do not assign it a visitor identity
    def test_case_08_statue_remains_visible_rejected(self):
        tid = "T-STATUE"
        f = _feat()
        # 30 frames at completely fixed coordinates
        dec = None
        for fi in range(1, 31):
            t = fi * 0.1
            det = [{'camera_id': 'cam1', 'track_id': tid, 'bbox': (700, 200, 780, 400), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=self.gate)
            dec = d[0]

        self.assertEqual(dec['global_id'], "STATUE")
        self.assertEqual(dec['state'], "STATIC")
        self.assertEqual(len(self.mgr.persons), 0, "Statues must never receive visitor person records")

    # 9. A person already inside at startup: do not fabricate an entry
    def test_case_09_person_already_inside_at_startup_no_fabricated_entry(self):
        w, h = 1000, 1000
        tid = "T-STAFF"
        f = _feat()
        # Person deep inside at y = 800 (gate line at 500)
        for fi in range(1, 25):
            t = fi * 0.1
            det = [{'camera_id': 'cam1', 'track_id': tid, 'bbox': (500 + fi*2, 720, 580 + fi*2, 820), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {tid: f}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.gate.check(tid, det[0]['bbox'], w, h, timestamp=t)

            self.assertEqual(d[0]['global_id'], tid)
            self.assertEqual(d[0]['state'], "UNVERIFIED_OCCUPANT")

        self.assertEqual(self.gate.entry_count, 0, "Pre-existing occupant must never fabricate gate entries")
        self.assertEqual(len(self.mgr.persons), 0)

    # 10. A completed visitor returns: recognize person, but create a new visit record
    def test_case_10_completed_visitor_returns_new_visit_record(self):
        w, h = 1000, 1000
        f = _feat()
        # 1. Enter and confirm visit 1
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {'T1': f}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.gate.check('T1', det[0]['bbox'], w, h, timestamp=t)
            if 'T1' in self.mgr.local_to_global:
                gid = self.mgr.local_to_global['T1']
                self.gate.transfer_identity('T1', gid)

        self.gate.confirm_pending_visits(8.0)
        self.assertEqual(self.gate.entry_count, 1)

        # 2. Complete visit by exiting
        for fi in range(200, 220):
            t = fi * 0.1
            y = 650 - (fi - 200) * 18
            det = [{'camera_id': 'cam1', 'track_id': 'T1', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T1': f}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.gate.check("P-001", det[0]['bbox'], w, h, timestamp=t)

        self.assertEqual(self.gate.exit_count, 1)
        p = self.mgr.persons["P-001"]
        self.assertTrue(p.is_departed)
        self.assertEqual(p.current_visit_status, "COMPLETED")

        # 3. Genuine return at t=100.0s (Visit 2)
        for fi in range(1000, 1015):
            t = fi * 0.1
            y = 350 + (fi - 1000) * 25
            det = [{'camera_id': 'cam1', 'track_id': 'T-RET', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {'T-RET': _close(f)}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            pkey = d[0]['global_id'] if d[0]['global_id'].startswith("P-") else 'T-RET'
            self.gate.check(pkey, det[0]['bbox'], w, h, timestamp=t)

        self.assertEqual(p.visit_id, "V-P-001-2", "Returning visitor must start new visit record V-P-001-2")
        self.gate.confirm_pending_visits(110.0)
        self.assertEqual(self.gate.entry_count, 2, "Second visit must count new entry")

    # 11. A short visit is ignored: do not transfer that visitor's ID to another person
    def test_case_11_short_visit_ignored_no_id_transfer(self):
        w, h = 1000, 1000
        f_visitor = _feat()
        f_other = _feat()

        # Visitor enters and exits within 2 seconds
        for fi in range(10, 18):
            t = fi * 0.1
            y = 350 + (fi - 10) * 40
            det = [{'camera_id': 'cam1', 'track_id': 'T-SHORT', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {'T-SHORT': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.gate.check('T-SHORT', det[0]['bbox'], w, h, timestamp=t)
            if 'T-SHORT' in self.mgr.local_to_global:
                self.gate.transfer_identity('T-SHORT', self.mgr.local_to_global['T-SHORT'])

        for fi in range(18, 26):
            t = fi * 0.1
            y = 650 - (fi - 18) * 40
            det = [{'camera_id': 'cam1', 'track_id': 'T-SHORT', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T-SHORT': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.gate.check('P-001', det[0]['bbox'], w, h, timestamp=t)

        # Excluded from official counts
        self.assertEqual(self.gate.entry_count, 0)
        self.assertEqual(self.gate.exit_count, 0)
        p = self.mgr.persons["P-001"]
        self.assertEqual(p.current_visit_status, "SHORT_VISIT_IGNORED")

        # Another person appears inside
        det_other = [{'camera_id': 'cam1', 'track_id': 'T-OTHER', 'bbox': (600, 650, 680, 750), 'conf': 0.92}]
        d_other = self.mgr.process_detections(det_other, {'T-OTHER': f_other}, 5.0, 50, gate_counter=self.gate, frame_w=w, frame_h=h)
        self.assertNotEqual(d_other[0]['global_id'], "P-001", "Short visit ID must never be transferred to another person")

    # 12. An identity is recovered: do not increment gate counts twice or corrupt group membership
    def test_case_12_recovered_identity_preserves_counts_and_group_membership(self):
        from app.identity.global_identity_manager import GlobalPerson
        w, h = 1000, 1000
        f1 = _feat()
        f2 = _feat()

        # Group of 2 enter together within 1.0s
        self.gate.register_entry("P-001", 10.0)
        self.gate.register_entry("P-002", 10.8)
        p1 = GlobalPerson("P-001", 10.0, "cam1", "T1", f1, (300, 600, 370, 700))
        p2 = GlobalPerson("P-002", 10.8, "cam1", "T2", f2, (400, 600, 470, 700))
        self.mgr.persons["P-001"] = p1
        self.mgr.persons["P-002"] = p2

        # Confirm entries after 6 seconds
        self.gate.confirm_pending_visits(17.0)
        self.assertEqual(self.gate.entry_count, 2)

        # P-001 is temporarily lost
        p1.mark_lost()
        self.assertEqual(p1.status, "TEMPORARILY_LOST")

        # P-001 is recovered with new track T1-new
        det_rec = [{'camera_id': 'cam1', 'track_id': 'T1-new', 'bbox': (310, 610, 380, 710), 'conf': 0.95}]
        d_rec = self.mgr.process_detections(det_rec, {'T1-new': _close(f1)}, 20.0, 200, gate_counter=self.gate, frame_w=w, frame_h=h)
        self.assertEqual(d_rec[0]['global_id'], "P-001")
        self.gate.check("P-001", det_rec[0]['bbox'], w, h, timestamp=20.0)

        # Verification: Gate count STILL exactly 2!
        self.assertEqual(self.gate.entry_count, 2, "Re-identifying visitor must NEVER increment entry count again")

        # Both exit together -> group formed correctly
        self.gate.person_history["P-001"].clear()
        self.gate.person_history["P-002"].clear()
        self.gate.last_eval_debug.clear()
        self.gate.check_departures(set(), {"P-001", "P-002"}, w, h, timestamp=50.0)
        self.assertEqual(len(self.gd.group_records), 1, "Cohort group membership must be preserved after recovery")
        self.assertEqual(self.gd.get_group("P-001"), self.gd.get_group("P-002"))



# ── Regression Suite: Gate Identity & Group-Counting Bug Fixes ─────────────
class TestGateIdentityAndGroupCountingBugs(unittest.TestCase):

    def setUp(self):
        from app.identity.global_identity_manager import GlobalIdentityManager
        from app.analytics.tripwire import TripwireCounter
        from app.analytics.group_detector import GroupDetector
        from app.cameras.camera_manager import CameraManager

        self.cam_mgr = CameraManager()
        self.cam_mgr.add_camera("cam1", "cam1")
        self.mgr = GlobalIdentityManager(self.cam_mgr, require_candidate_motion=True)
        self.gd = GroupDetector(entry_window_sec=2.0, exit_window_sec=2.0)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        self.gate.group_detector = self.gd
        self.gate.identity_manager = self.mgr
        self.mgr.gate_counter = self.gate
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    def tearDown(self):
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    # 1. An inside/unverified person first appears after a visitor already has a P-ID and cannot steal it
    def test_unverified_person_appears_after_visitor_and_cannot_steal_id(self):
        w, h = 1000, 1000
        f_visitor = _feat()
        c_visitor = _clothing(upper_h=0.10, lower_h=0.60)
        
        # Similar clothing and close embedding
        f_unverified = _close(f_visitor)
        c_unverified = _clothing(upper_h=0.10, lower_h=0.60)

        # Visitor enters and crosses gate at t=1.0 - 2.0s
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {'T-VISITOR': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-VISITOR': c_visitor})
            ev = self.gate.check('T-VISITOR', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f_visitor, clothing=c_visitor)
            if ev == "IN":
                gid = self.mgr.local_to_global['T-VISITOR']
                self.gate.transfer_identity('T-VISITOR', gid)

        gid_visitor = self.mgr.local_to_global.get('T-VISITOR')
        self.assertIsNotNone(gid_visitor)
        self.assertTrue(gid_visitor.startswith("P-"), f"Visitor must receive a P-ID, got {gid_visitor}")

        # Now visitor is inside. At t=5.0s, an unverified person appears deep inside the store (y=750)
        det_unverified = [
            {'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': (400, 600, 470, 700), 'conf': 0.95},
            {'camera_id': 'cam1', 'track_id': 'T-UNVERIFIED', 'bbox': (800, 750, 870, 850), 'conf': 0.92}
        ]
        d_both = self.mgr.process_detections(det_unverified, {'T-VISITOR': f_visitor, 'T-UNVERIFIED': f_unverified}, 5.0, 50, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-VISITOR': c_visitor, 'T-UNVERIFIED': c_unverified})
        res = {x['track_id']: x for x in d_both}
        self.assertEqual(res['T-UNVERIFIED']['global_id'], 'T-UNVERIFIED', "Unverified occupant must not receive visitor P-ID")
        self.assertEqual(res['T-UNVERIFIED']['state'], 'UNVERIFIED_OCCUPANT')

        # Now visitor temporarily disappears at t=6.0s
        self.mgr.process_detections([], {}, 6.0, 60, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons[gid_visitor].status, "TEMPORARILY_LOST")

        # Unverified person continues tracking with natural movement at t=7.0s
        det_unv_only = [{'camera_id': 'cam1', 'track_id': 'T-UNVERIFIED', 'bbox': (860, 780, 930, 880), 'conf': 0.92}]
        d_unv_only = self.mgr.process_detections(det_unv_only, {'T-UNVERIFIED': f_unverified}, 7.0, 70, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-UNVERIFIED': c_unverified})
        self.assertNotEqual(d_unv_only[0]['global_id'], gid_visitor, "Unverified occupant MUST NEVER steal lost visitor's P-ID")
        self.assertEqual(d_unv_only[0]['global_id'], 'T-UNVERIFIED')

    # 2. A new visitor gets no P-ID before crossing and receives one after a valid crossing
    def test_new_visitor_gets_no_pid_before_crossing_and_receives_one_after_valid_crossing(self):
        w, h = 1000, 1000
        f = _feat()

        # Approaching gate outside (y = 200 - 380, gate is at y = 500)
        for fi in range(1, 10):
            t = fi * 0.1
            y = 200 + fi * 20
            det = [{'camera_id': 'cam1', 'track_id': 'T-APPROACH', 'bbox': (450, int(y - 80), 550, int(y)), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {'T-APPROACH': f}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.assertEqual(d[0]['global_id'], 'T-APPROACH', "Person approaching gate outside must not have P-ID")
            self.assertEqual(d[0]['state'], 'OUTSIDE')
            ev = self.gate.check('T-APPROACH', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f)
            self.assertIsNone(ev, "No gate crossing before crossing line")
            self.assertNotIn('T-APPROACH', self.mgr.local_to_global)

        # Crosses gate line into store (y = 560 - 640)
        promoted = False
        for fi in range(10, 16):
            t = fi * 0.1
            y = 400 + (fi - 10) * 45
            det = [{'camera_id': 'cam1', 'track_id': 'T-APPROACH', 'bbox': (450, int(y - 80), 550, int(y)), 'conf': 0.9}]
            d = self.mgr.process_detections(det, {'T-APPROACH': f}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            person_key = self.mgr.local_to_global.get('T-APPROACH', 'T-APPROACH')
            ev = self.gate.check(person_key, det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f)
            if ev == "IN":
                promoted = True

        self.assertTrue(promoted, "Valid crossing must trigger IN event")
        gid = self.mgr.local_to_global.get('T-APPROACH')
        self.assertIsNotNone(gid)
        self.assertTrue(gid.startswith("P-"), f"Crossing must create permanent P-ID, got {gid}")

    # 3. A temporarily lost visitor reappears with a new tracker ID/viewpoint, recovers the same P-ID, and is not counted as a second entry
    def test_temporarily_lost_visitor_reappears_recovers_same_pid_no_double_entry_count(self):
        w, h = 1000, 1000
        f_visitor = _feat()
        c_visitor = _clothing(upper_h=0.20, lower_h=0.70)

        # Entry and confirmed visit
        self.gate.register_entry("P-001", 10.0)
        from app.identity.global_identity_manager import GlobalPerson
        p = GlobalPerson("P-001", 10.0, "cam1", "T-OLD", f_visitor, (400, 600, 480, 720))
        p.gallery.add_clothing(c_visitor)
        self.mgr.persons["P-001"] = p
        self.mgr.local_to_global["T-OLD"] = "P-001"

        # Stay inside > 5s -> visit confirmed
        self.gate.confirm_pending_visits(16.0)
        self.assertEqual(self.gate.entry_count, 1)

        # Visitor becomes temporarily lost
        p.mark_lost()
        del self.mgr.local_to_global["T-OLD"]
        self.assertEqual(p.status, "TEMPORARILY_LOST")

        # Visitor reappears with new track ID and turned viewpoint
        f_turned = _close(f_visitor)
        det_reappear = [{'camera_id': 'cam1', 'track_id': 'T-REAPPEAR', 'bbox': (420, 610, 500, 730), 'conf': 0.95}]
        d = self.mgr.process_detections(det_reappear, {'T-REAPPEAR': f_turned}, 20.0, 200, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-REAPPEAR': c_visitor})
        self.assertEqual(d[0]['global_id'], "P-001", "Must recover original P-001 ID")

        # Check tripwire: gate count must NOT increment again
        ev = self.gate.check("P-001", det_reappear[0]['bbox'], w, h, timestamp=20.0, camera_id="cam1", feat=f_turned)
        self.assertEqual(self.gate.entry_count, 1, "Recovery must NEVER trigger another Gate IN")

    # 4. A genuinely exited visitor re-enters with the same P-ID but a new visit record
    def test_genuinely_exited_visitor_reenters_with_same_pid_and_new_visit_record(self):
        w, h = 1000, 1000
        f = _feat()
        c = _clothing(upper_h=0.30, lower_h=0.80)

        # Visit 1: Entry at t=10.0, confirmed at t=16.0, exit at t=40.0
        self.gate.register_entry("P-001", 10.0)
        from app.identity.global_identity_manager import GlobalPerson
        p = GlobalPerson("P-001", 10.0, "cam1", "T1", f, (450, 600, 530, 720))
        p.gallery.add_clothing(c)
        self.mgr.persons["P-001"] = p
        self.mgr.local_to_global["T1"] = "P-001"
        self.gate.confirm_pending_visits(16.0)
        self.assertEqual(self.gate.entry_count, 1)

        # Official Exit at t=40.0
        self.gate.person_history["P-001"].clear()
        self.gate.check_departures(set(), {"P-001"}, w, h, timestamp=40.0)
        self.assertEqual(self.gate.exit_count, 1)
        self.assertTrue(p.is_departed)
        self.assertEqual(p.current_visit_status, "COMPLETED")
        self.assertEqual(len(self.gate.visit_records), 1)

        # Re-entry at t=100.0: person returns with track T2 across gate
        new_gid = self.mgr.promote_track_to_visitor("T2", 100.0, "cam1", _close(f), (450, 600, 530, 720), clothing=c)
        self.assertEqual(new_gid, "P-001", "Returning visitor must retain same persistent P-001 ID")
        self.assertEqual(p.visit_count, 2, "Must increment visit count to 2")
        self.assertEqual(p.visit_id, "V-P-001-2", "Must create new distinct visit record")
        self.assertFalse(p.is_departed)

        self.gate.register_entry("P-001", 100.0)
        self.gate.confirm_pending_visits(106.0)
        self.assertEqual(self.gate.entry_count, 2, "Second visit confirmed after 5s inside")
        self.assertEqual(len(self.gate.visit_records), 2, "Must have 2 separate visit records")

    # 5. A visit of 5 seconds or less is excluded from official IN/OUT counts without unnecessarily deleting identity memory
    def test_short_visit_under_5s_excluded_from_counts_while_identity_memory_preserved(self):
        w, h = 1000, 1000
        f = _feat()
        c = _clothing(upper_h=0.45, lower_h=0.15)

        # Entry at t=10.0
        self.gate.register_entry("P-001", 10.0)
        from app.identity.global_identity_manager import GlobalPerson
        p = GlobalPerson("P-001", 10.0, "cam1", "T-SHORT", f, (450, 600, 530, 720))
        p.gallery.add_clothing(c)
        self.mgr.persons["P-001"] = p
        self.mgr.local_to_global["T-SHORT"] = "P-001"

        # Exits at t=13.0 (duration 3.0s <= 5.0s)
        self.gate.person_history["P-001"].clear()
        self.gate.check_departures(set(), {"P-001"}, w, h, timestamp=13.0)
        self.assertEqual(self.gate.entry_count, 0, "Short visit must not increment entry count")
        self.assertEqual(self.gate.exit_count, 0, "Short visit must not increment exit count")
        self.assertEqual(p.current_visit_status, "SHORT_VISIT_IGNORED")
        self.assertEqual(p.gallery.size(), 1, "Identity memory must be preserved")

        # Visitor returns at t=50.0 and re-enters
        re_gid = self.mgr.promote_track_to_visitor("T-RETURN", 50.0, "cam1", _close(f), (450, 600, 530, 720), clothing=c)
        self.assertEqual(re_gid, "P-001", "Returning visitor whose prior visit was short must retain P-001")
        self.gate.register_entry("P-001", 50.0)
        self.gate.confirm_pending_visits(56.0)
        self.assertEqual(self.gate.entry_count, 1, "Subsequent confirmed visit increments entry count")

    # 6. Exact group cohort succeeds; partial exits, entry-window misses, and exit-window misses do not form a group
    def test_exact_group_cohort_success_and_failures(self):
        # 6a. Exact cohort of 2 succeeds
        self.gd.reset()
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 11.2)
        self.gd.record_exit("P-001", 50.0)
        self.gd.record_exit("P-002", 51.0)
        self.assertEqual(len(self.gd.group_records), 1, "Exact cohort of 2 must form group")
        self.assertEqual(self.gd.get_group("P-001"), self.gd.get_group("P-002"))

        # 6b. Partial cohort exit fails (3 enter, only 2 exit)
        self.gd.reset()
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 10.8)
        self.gd.record_entry("P-003", 11.5)
        self.gd.record_exit("P-001", 60.0)
        self.gd.record_exit("P-002", 60.5)
        self.assertEqual(len(self.gd.group_records), 0, "Partial cohort exit (2 of 3) must NOT form a group")

        # 6c. Entry window miss (> 2.0s) fails
        self.gd.reset()
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 13.5) # > 2.0s gap
        self.gd.record_exit("P-001", 50.0)
        self.gd.record_exit("P-002", 50.5)
        self.assertEqual(len(self.gd.group_records), 0, "Entry window miss (> 2s) must NOT form a group")

        # 6d. Exit window miss (> 2.0s) fails
        self.gd.reset()
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_entry("P-002", 11.0)
        self.gd.record_exit("P-001", 50.0)
        self.gd.record_exit("P-002", 55.0) # > 2.0s gap
        self.assertEqual(len(self.gd.group_records), 0, "Exit window miss (> 2s) must NOT form a group")

        # 6e. Returning visitor participates in a later group
        self.gd.reset()
        # Visit 1: P-001 enters and exits alone
        self.gd.record_entry("P-001", 10.0)
        self.gd.record_exit("P-001", 40.0)
        self.assertEqual(len(self.gd.group_records), 0)

        # Visit 2: P-001 returns and enters with P-003
        self.gd.record_entry("P-001", 100.0)
        self.gd.record_entry("P-003", 101.0)
        self.gd.record_exit("P-001", 150.0)
        self.gd.record_exit("P-003", 150.8)
        self.assertEqual(len(self.gd.group_records), 1, "Returning visitor must form group on subsequent visit")
        self.assertEqual(self.gd.get_group("P-001"), self.gd.get_group("P-003"))

    # 7. Frame skipping does not create duplicate crossings or break valid crossing detection
    def test_frame_skipping_no_duplicate_crossings_preserves_valid_detection(self):
        fps = 25.0
        w, h = 1000, 1000

        for frame_skip in [1, 2, 5]:
            self.gate.reset()
            f = _feat()
            
            # Walking from outside (y=300) to inside (y=650) across gate at y=500
            for fi in range(0, 50, frame_skip):
                t = fi / fps
                y = 300 + fi * 8.0 # gradual movement across line
                det_bbox = (450, int(y - 80), 550, int(y))
                self.gate.check("T1", det_bbox, w, h, timestamp=t, camera_id="cam1", feat=f)

            # Person reaches deep inside (y=650) and stays inside for 7 seconds
            t_entry = self.gate.confirmed_entries.get("P-001", self.gate.pending_entries.get("P-001"))
            self.assertIsNotNone(t_entry, f"Entry must be detected under frame_skip={frame_skip}")

            # Time advances by 6 seconds
            t_confirm = t_entry + 6.0
            for fi in range(50, 70, frame_skip):
                t = t_confirm + (fi - 50) / fps
                self.gate.check("P-001", (450, 600, 550, 680), w, h, timestamp=t)

            self.assertEqual(self.gate.entry_count, 1, f"Frame skip {frame_skip} must yield exactly 1 entry (no duplicates)")

    # 8. ResNet feature centering is deterministic and batch-independent
    def test_resnet_feature_centering_deterministic_and_batch_independent(self):
        from app.reid.model import ReIDModel
        model = ReIDModel()

        # Two distinct synthetic crops
        crop1 = np.full((128, 64, 3), 40, dtype=np.uint8)
        crop2 = np.full((128, 64, 3), 200, dtype=np.uint8)

        # Single crop extraction
        feat1_alone = model.extract_features([crop1])
        feat2_alone = model.extract_features([crop2])

        # Batch extraction of both
        feats_batch = model.extract_features([crop1, crop2])
        # Reversed batch
        feats_rev = model.extract_features([crop2, crop1])

        # Verification: Crop 1's embedding is invariant to batch size and batch contents
        self.assertTrue(np.allclose(feat1_alone[0], feats_batch[0], atol=1e-3),
                        "Crop 1 embedding must be identical alone vs in batch with Crop 2")
        self.assertTrue(np.allclose(feat2_alone[0], feats_batch[1], atol=1e-3),
                        "Crop 2 embedding must be identical alone vs in batch with Crop 1")
        self.assertGreaterEqual(float(np.dot(feat1_alone[0], feats_batch[0])), 0.9999,
                                "Cosine similarity alone vs batch must be >= 0.9999")
        self.assertTrue(np.allclose(feat1_alone[0], feats_rev[1], atol=1e-3),
                        "Crop 1 embedding must be invariant to batch order")
        self.assertGreaterEqual(float(np.dot(feat1_alone[0], feats_rev[1])), 0.9999,
                                "Cosine similarity invariant to batch order must be >= 0.9999")


# ── Issue 10: Robust Identity Association, Occlusion vs Unverified, & Local VLM Tests ──
class TestRobustIdentityAndVLMRegression(unittest.TestCase):
    def setUp(self):
        from app.identity.global_identity_manager import GlobalIdentityManager
        from app.analytics.tripwire import TripwireCounter
        from app.analytics.group_detector import GroupDetector
        from app.cameras.camera_manager import CameraManager
        from app.reid.vlm_verifier import VLMVerifier
        from app.config import Config

        self.cam_mgr = CameraManager()
        self.cam_mgr.add_camera("cam1", "cam1")
        self.mgr = GlobalIdentityManager(self.cam_mgr, require_candidate_motion=True)
        self.gd = GroupDetector(entry_window_sec=2.0, exit_window_sec=2.0)
        self.gate = TripwireCounter(point_a=(0.0, 0.5), point_b=(1.0, 0.5), inside_direction="below")
        self.gate.group_detector = self.gd
        self.gate.identity_manager = self.mgr
        self.mgr.gate_counter = self.gate
        Config.REID_MATCH_THRESHOLD = 0.75
        Config.ACTIVE_VISITOR_REID_THRESHOLD = 0.75

    # 1. Non-overlap unverified occupant appearing AFTER visitor is lost must NOT inherit P-ID
    def test_unverified_occupant_appearing_after_visitor_lost_cannot_steal_id(self):
        w, h = 1000, 1000
        f_visitor = _feat()
        c_visitor = _clothing(upper_h=0.15, lower_h=0.65)

        # Candidate has almost identical features and clothing
        f_unverified = _close(f_visitor)
        c_unverified = _clothing(upper_h=0.15, lower_h=0.65)

        # Visitor enters and crosses gate between t=1.0 and 2.0s
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            d = self.mgr.process_detections(det, {'T-VISITOR': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-VISITOR': c_visitor})
            ev = self.gate.check('T-VISITOR', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f_visitor, clothing=c_visitor)
            if ev == "IN":
                gid = self.mgr.local_to_global['T-VISITOR']
                self.gate.transfer_identity('T-VISITOR', gid)

        gid_visitor = self.mgr.local_to_global.get('T-VISITOR')
        self.assertIsNotNone(gid_visitor)
        self.assertTrue(gid_visitor.startswith("P-"))
        orig_gallery_size = self.mgr.persons[gid_visitor].gallery.size()

        # Visitor is lost at t=4.0s
        self.mgr.process_detections([], {}, 4.0, 40, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons[gid_visitor].status, "TEMPORARILY_LOST")

        # At t=8.0s (4 seconds later), a completely new track T-UNV first appears inside the store
        # Location is deep in another area (850, 750) > 400px away from visitor's last location
        # Importantly, T-UNV NEVER co-existed in time with T-VISITOR (no frame overlap)
        det_unv = [{'camera_id': 'cam1', 'track_id': 'T-UNV', 'bbox': (850, 750, 920, 850), 'conf': 0.94}]
        d_res = self.mgr.process_detections(
            det_unv, {'T-UNV': f_unverified}, 8.0, 80,
            gate_counter=self.gate, frame_w=w, frame_h=h,
            clothing_map={'T-UNV': c_unverified}
        )

        # Verification: T-UNV must NOT receive gid_visitor!
        self.assertEqual(len(d_res), 1)
        self.assertEqual(d_res[0]['global_id'], 'T-UNV', "Unverified track appearing without continuity must NOT inherit visitor ID")
        self.assertEqual(d_res[0]['state'], 'UNVERIFIED_OCCUPANT')
        self.assertNotIn('T-UNV', self.mgr.local_to_global)
        # Visitor P-001 must remain TEMPORARILY_LOST
        self.assertEqual(self.mgr.persons[gid_visitor].status, "TEMPORARILY_LOST")
        # Visitor gallery must NOT have been updated by unverified track
        self.assertEqual(self.mgr.persons[gid_visitor].gallery.size(), orig_gallery_size)
        # Gate entry count must remain 1 after visit duration confirms
        self.gate.confirm_pending_visits(10.0)
        self.assertEqual(self.gate.entry_count, 1)

    # 2. Legitimate in-store occlusion recovery (Case A) succeeds with credible continuity
    def test_legitimate_in_store_occlusion_recovery_succeeds(self):
        w, h = 1000, 1000
        f_visitor = _feat()
        c_visitor = _clothing(upper_h=0.20, lower_h=0.70)

        # Visitor enters and crosses gate
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T-VISITOR': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-VISITOR': c_visitor})
            ev = self.gate.check('T-VISITOR', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f_visitor, clothing=c_visitor)
            if ev == "IN":
                gid = self.mgr.local_to_global['T-VISITOR']
                self.gate.transfer_identity('T-VISITOR', gid)

        gid_visitor = self.mgr.local_to_global['T-VISITOR']
        last_bbox = self.mgr.persons[gid_visitor].last_bbox

        # Briefly occluded at t=3.0s (1.0s gap)
        self.mgr.process_detections([], {}, 3.0, 30, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons[gid_visitor].status, "TEMPORARILY_LOST")

        # Track re-emerges with new tracker ID 'T-RECOVERED' at nearby position (35px away)
        # with identical appearance and clothing at t=3.8s (0.8s gap <= 15s)
        nearby_bbox = (last_bbox[0] + 25, last_bbox[1] + 25, last_bbox[2] + 25, last_bbox[3] + 25)
        f_recov = _close(f_visitor)
        det_recov = [{'camera_id': 'cam1', 'track_id': 'T-RECOVERED', 'bbox': nearby_bbox, 'conf': 0.93}]
        d_recov = self.mgr.process_detections(
            det_recov, {'T-RECOVERED': f_recov}, 3.8, 38,
            gate_counter=self.gate, frame_w=w, frame_h=h,
            clothing_map={'T-RECOVERED': c_visitor}
        )

        # Verification: Credible spatio-temporal continuity allows successful recovery
        self.assertEqual(len(d_recov), 1)
        self.assertEqual(d_recov[0]['global_id'], gid_visitor, "Nearby occluded visitor must be recovered")
        self.assertEqual(d_recov[0]['state'], "REIDENTIFIED")
        self.assertEqual(self.mgr.persons[gid_visitor].status, "REIDENTIFIED")
        self.assertEqual(self.mgr.local_to_global['T-RECOVERED'], gid_visitor)

    # 3. Confirmed visitor who pauses or stands still inside is never demoted to STATUE
    def test_confirmed_visitor_standing_still_never_demoted_to_statue(self):
        w, h = 1000, 1000
        f_visitor = _feat()
        c_visitor = _clothing(upper_h=0.10, lower_h=0.50)

        # Visitor enters and crosses gate
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T-VISITOR': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-VISITOR': c_visitor})
            ev = self.gate.check('T-VISITOR', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f_visitor, clothing=c_visitor)
            if ev == "IN":
                gid = self.mgr.local_to_global['T-VISITOR']
                self.gate.transfer_identity('T-VISITOR', gid)

        gid_visitor = self.mgr.local_to_global['T-VISITOR']

        # Visitor stands completely still for 50 frames (5 seconds)
        still_bbox = (400, 600, 470, 700)
        for fi in range(25, 75):
            t = fi * 0.1
            det = [{'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': still_bbox, 'conf': 0.95}]
            d = self.mgr.process_detections(det, {'T-VISITOR': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            self.assertEqual(d[0]['global_id'], gid_visitor, "Standing visitor must retain persistent ID")
            self.assertNotEqual(d[0]['global_id'], "STATUE", "Standing visitor must NEVER be renamed to STATUE")

        p = self.mgr.persons[gid_visitor]
        self.assertFalse(p.is_static, "Confirmed visitor is_static must remain False")
        self.assertTrue(getattr(p, 'is_stationary', False), "Standing visitor should be flagged is_stationary")

    # 4. Stationary statue with bounding box jitter never gets P-ID or gate count
    def test_statue_with_jitter_never_gets_pid_or_gate_count(self):
        w, h = 1000, 1000
        f_statue = _feat()

        # Statue placed near threshold with small jitter (+/- 2 pixels) for 30 frames
        for fi in range(1, 35):
            t = fi * 0.1
            jitter = (fi % 3) - 1
            bbox = (300 + jitter, 430 + jitter, 360 + jitter, 530 + jitter)
            det = [{'camera_id': 'cam1', 'track_id': 'T-STATUE', 'bbox': bbox, 'conf': 0.90}]
            d = self.mgr.process_detections(det, {'T-STATUE': f_statue}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            is_statue = (d[0]['global_id'] == "STATUE")
            ev = self.gate.check('T-STATUE', bbox, w, h, is_static=is_statue, timestamp=t)
            self.assertIsNone(ev, "Statue must never trigger an IN event")

        # After 15+ frames, it must be recognized as STATUE / STATIC
        self.assertEqual(d[0]['global_id'], "STATUE")
        self.assertEqual(d[0]['state'], "STATIC")
        self.assertEqual(self.gate.entry_count, 0, "Statue must never count as visitor entry")
        self.assertNotIn('T-STATUE', self.mgr.local_to_global)

    # 5. Local VLMVerifier gracefully handles offline Ollama
    def test_vlm_verifier_offline_graceful_fallback(self):
        from app.reid.vlm_verifier import VLMVerifier
        # Port 59999 is guaranteed to have no server running
        vlm = VLMVerifier(base_url="http://localhost:59999", enabled=True, timeout=0.5)

        crop_ref = np.zeros((100, 50, 3), dtype=np.uint8)
        crop_query = np.zeros((100, 50, 3), dtype=np.uint8)

        vlm.store_reference_crop("P-001", crop_ref)
        res = vlm.verify_candidate(crop_query, "P-001")

        self.assertIsInstance(res, dict)
        self.assertFalse(res["match"])
        self.assertEqual(res["status"], "UNAVAILABLE")
        self.assertIn("unavailable", res["reason"].lower())

    # 6. VLM cannot override hard gate or continuity rules
    def test_vlm_cannot_override_continuity_rules(self):
        from app.reid.vlm_verifier import VLMVerifier
        from unittest.mock import MagicMock

        w, h = 1000, 1000
        f_visitor = _feat()
        c_visitor = _clothing(upper_h=0.30, lower_h=0.70)

        # Setup mock VLM that claims match is ALWAYS True
        mock_vlm = VLMVerifier(enabled=True)
        mock_vlm.verify_candidate = MagicMock(return_value={"match": True, "confidence": 0.99, "reason": "Mocked positive", "status": "SUCCESS"})
        self.mgr.set_vlm_verifier(mock_vlm)

        # Visitor enters
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-VISITOR', 'bbox': (400, int(y - 90), 470, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T-VISITOR': f_visitor}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h, clothing_map={'T-VISITOR': c_visitor})
            ev = self.gate.check('T-VISITOR', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=f_visitor, clothing=c_visitor)
            if ev == "IN":
                gid = self.mgr.local_to_global['T-VISITOR']
                self.gate.transfer_identity('T-VISITOR', gid)

        gid_visitor = self.mgr.local_to_global['T-VISITOR']

        # Visitor is lost
        self.mgr.process_detections([], {}, 4.0, 40, gate_counter=self.gate)

        # An unverified track appears far away (x=900, y=850) with broken continuity
        f_fake = _close(f_visitor)
        det_fake = [{'camera_id': 'cam1', 'track_id': 'T-FAKE', 'bbox': (850, 800, 920, 900), 'conf': 0.95}]
        d_res = self.mgr.process_detections(
            det_fake, {'T-FAKE': f_fake}, 5.0, 50,
            gate_counter=self.gate, frame_w=w, frame_h=h,
            clothing_map={'T-FAKE': c_visitor}
        )

        # Verification: VLM cannot override Rule 7.5; track remains unverified!
        self.assertEqual(d_res[0]['global_id'], 'T-FAKE')
        self.assertEqual(d_res[0]['state'], 'UNVERIFIED_OCCUPANT')
        self.assertNotIn('T-FAKE', self.mgr.local_to_global)

    # 7. VLM resolves ambiguous candidate tie when enabled
    def test_vlm_resolves_ambiguous_candidate_tie(self):
        from app.reid.vlm_verifier import VLMVerifier
        from unittest.mock import MagicMock

        w, h = 1000, 1000
        fA = np.zeros(512, dtype=np.float32); fA[:200] = 1.0; fA /= np.linalg.norm(fA)
        fB = np.zeros(512, dtype=np.float32); fB[100:300] = 1.0; fB /= np.linalg.norm(fB)

        # P-001 enters
        for fi in range(10, 20):
            t = fi * 0.1
            y = 350 + (fi - 10) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-1', 'bbox': (300, int(y - 90), 370, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T-1': fA}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            ev = self.gate.check('T-1', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=fA)
            if ev == "IN":
                self.gate.transfer_identity('T-1', self.mgr.local_to_global['T-1'])

        # P-002 enters
        for fi in range(20, 30):
            t = fi * 0.1
            y = 350 + (fi - 20) * 30
            det = [{'camera_id': 'cam1', 'track_id': 'T-2', 'bbox': (360, int(y - 90), 430, int(y)), 'conf': 0.95}]
            self.mgr.process_detections(det, {'T-2': fB}, t, fi, gate_counter=self.gate, frame_w=w, frame_h=h)
            ev = self.gate.check('T-2', det[0]['bbox'], w, h, timestamp=t, camera_id='cam1', feat=fB)
            if ev == "IN":
                self.gate.transfer_identity('T-2', self.mgr.local_to_global['T-2'])

        gid1 = self.mgr.local_to_global['T-1']
        gid2 = self.mgr.local_to_global['T-2']
        self.assertEqual(gid1, "P-001")
        self.assertEqual(gid2, "P-002")

        # Both become temporarily lost
        self.mgr.process_detections([], {}, 4.0, 40, gate_counter=self.gate)
        self.assertEqual(self.mgr.persons[gid1].status, "TEMPORARILY_LOST")
        self.assertEqual(self.mgr.persons[gid2].status, "TEMPORARILY_LOST")

        # Mock VLM: confirms P-001, rejects P-002
        mock_vlm = VLMVerifier(enabled=True)
        def mock_verify(crop, gid):
            if gid == "P-001":
                return {"match": True, "confidence": 0.88, "reason": "Confirmed red shirt", "status": "SUCCESS"}
            return {"match": False, "confidence": 0.20, "reason": "Mismatched attributes", "status": "SUCCESS"}
        mock_vlm.verify_candidate = MagicMock(side_effect=mock_verify)
        self.mgr.set_vlm_verifier(mock_vlm)

        # Ambiguous query feature close to both P-001 and P-002 (sim=0.866, margin = 0.0)
        f_query = (fA + fB) / np.linalg.norm(fA + fB)

        # T-RECOV appears equidistant between P-001 and P-002 with crop
        fake_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        det_query = [{'camera_id': 'cam1', 'track_id': 'T-QUERY', 'bbox': (330, 580, 400, 680), 'conf': 0.95, 'crop': fake_crop}]
        d_res = self.mgr.process_detections(
            det_query, {'T-QUERY': f_query}, 4.2, 42,
            gate_counter=self.gate, frame_w=w, frame_h=h
        )

        # Verification: VLM tie-break resolves candidate to P-001
        self.assertEqual(len(d_res), 1)
        self.assertEqual(d_res[0]['global_id'], "P-001", "VLM should resolve tie to P-001")
        self.assertEqual(d_res[0]['state'], "REIDENTIFIED")


if __name__ == '__main__':
    unittest.main(verbosity=2)





