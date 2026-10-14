import sys
import os
import cv2
import threading
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QCheckBox, QGroupBox, QSlider, QSpinBox
)
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtGui import QImage, QPixmap

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.video.processor import MultiCameraProcessor
from app.config import Config
from app.analytics.tripwire import TripwireCounter
from frontend.interactive_canvas import InteractiveVideoCanvas


class VideoProcessorThread(QObject):
    frame_ready      = Signal(str, QImage)
    stats_updated    = Signal(int, int, int, int, int, int, int)  # active, unique, lost, in, out, statues, groups
    progress_updated = Signal(int)
    finished         = Signal()

    def __init__(self, cameras, roi_norm, gate_norm, gate_dir="below", show_statues=True, frame_skip=1):
        super().__init__()
        self.cameras      = cameras
        self.roi_norm     = roi_norm      # (left, top, right, bottom) normalized
        self.gate_norm    = gate_norm     # ((ax, ay), (bx, by)) normalized
        self.gate_dir     = gate_dir      # "below" or "above"
        self.show_statues = show_statues  # whether to draw dimmed statue box
        self.frame_skip   = max(1, int(frame_skip))

        self.processor = None
        self.running   = False
        self.paused    = False
        self.tripwire  = None

    def update_gate_geometry(self, p1, p2, inside_dir=None):
        self.gate_norm = (p1, p2)
        if inside_dir:
            self.gate_dir = inside_dir
        if self.tripwire:
            self.tripwire.update_geometry(p1, p2, inside_dir)

    def update_roi(self, l, t, r, b):
        self.roi_norm = (l, t, r, b)

    def set_vlm_enabled(self, enabled):
        if getattr(self, "vlm_verifier", None):
            self.vlm_verifier.enabled = enabled

    def run(self):
        self.processor = MultiCameraProcessor()
        for cam_id, path in self.cameras:
            self.processor.add_camera_stream(cam_id, path)

        # Initialize Gate Tripwire
        p1, p2 = self.gate_norm
        self.tripwire = TripwireCounter(point_a=p1, point_b=p2, inside_direction=self.gate_dir)
        self.tripwire.group_detector = self.processor.group_detector
        self.tripwire.identity_manager = self.processor.identity_manager
        self.processor.identity_manager.gate_counter = self.tripwire

        # Local VLM Verifier (Ollama)
        from app.reid.vlm_verifier import VLMVerifier
        self.vlm_verifier = VLMVerifier(
            base_url=getattr(Config, "VLM_BASE_URL", "http://localhost:11434"),
            model=getattr(Config, "VLM_MODEL", "qwen3.5:2b-q4_K_M"),
            enabled=getattr(Config, "VLM_ENABLED", True),
            timeout=getattr(Config, "VLM_TIMEOUT", 3.0),
        )
        self.processor.identity_manager.set_vlm_verifier(self.vlm_verifier)

        self.running  = True
        frame_idx     = 0
        id_mgr        = self.processor.identity_manager

        total_frames = 100
        if self.processor.video_caps:
            first_cap = list(self.processor.video_caps.values())[0]
            total_frames = int(first_cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            if total_frames <= 0:
                total_frames = 100

        while self.running:
            if self.paused:
                import time
                time.sleep(0.1)
                continue

            all_frames_empty = True

            # Dynamic live ROI from UI
            roi_l, roi_t, roi_r, roi_b = self.roi_norm

            for cam_id, cap in self.processor.video_caps.items():
                ret, frame = cap.read()
                if not ret:
                    continue

                all_frames_empty = False
                current_frame_idx = frame_idx

                if current_frame_idx % self.frame_skip != 0:
                    continue

                frame_h, frame_w = frame.shape[:2]

                fps       = cap.get(cv2.CAP_PROP_FPS) or 25
                timestamp = current_frame_idx / fps

                # Convert normalized ROI to pixel boundaries
                rx1 = int(roi_l * frame_w)
                ry1 = int(roi_t * frame_h)
                rx2 = int(roi_r * frame_w)
                ry2 = int(roi_b * frame_h)

                # ── 1. Local Tracking ──────────────────────────────────────
                raw_detections = self.processor.trackers[cam_id].process_frame(frame)

                # ── Keep all detections for global co-presence, candidate motion, and unverified tracking ──
                # Ensure ROI filtering does not discard observations needed to establish co-visibility
                detections = raw_detections

                # ── 2. Selective Feature Extraction (rolling gallery) ──────
                features_map     = {}
                crops_to_extract = []
                ids_to_extract   = []

                for d in detections:
                    tid = d['track_id']
                    bw  = d['bbox'][2] - d['bbox'][0]
                    bh  = d['bbox'][3] - d['bbox'][1]
                    if bw < Config.MIN_CROP_WIDTH or bh < Config.MIN_CROP_HEIGHT:
                        continue

                    extract = False
                    if tid not in id_mgr.local_to_global:
                        extract = True          # Unknown → must Re-ID
                    else:
                        gid = id_mgr.local_to_global[tid]
                        p   = id_mgr.persons.get(gid)
                        if p and (p.gallery.is_empty()
                                or p.gallery.size() < 3
                                or current_frame_idx % Config.GALLERY_UPDATE_INTERVAL == 0):
                            extract = True

                    if extract:
                        crops_to_extract.append(d['crop'])
                        ids_to_extract.append(tid)

                if crops_to_extract:
                    feats = self.processor.reid_model.extract_features(crops_to_extract)
                    for tid, feat in zip(ids_to_extract, feats):
                        features_map[tid] = feat

                # ── 3. Global Identity Management (with Gate Admission & Statue Rejection) ──
                decisions = id_mgr.process_detections(
                    detections, features_map, timestamp, current_frame_idx,
                    gate_counter=self.tripwire, frame_w=frame_w, frame_h=frame_h
                )
                dec_map = {d['track_id']: d for d in decisions}

                # ── 3.2 Gate Tripwire Line Crossing Check ──────────────────
                active_gids = set()
                for det in detections:
                    tid = det['track_id']
                    if tid not in dec_map:
                        continue
                    dec = dec_map[tid]
                    gid = dec['global_id']
                    state = dec['state']

                    # Statue / mannequin filter: Unpromoted static objects NEVER count as gate crossings!
                    is_static_person = (gid == "STATUE")
                    
                    # If this track is already a promoted visitor, use gid; else use tid
                    if gid and gid.startswith("P-"):
                        self.tripwire.transfer_identity(tid, gid)
                        person_key = gid
                    else:
                        person_key = tid

                    event = self.tripwire.check(
                        person_key, det['bbox'], frame_w, frame_h,
                        is_static=is_static_person, timestamp=timestamp,
                        camera_id=cam_id, feat=features_map.get(tid),
                        crop=det.get('crop')
                    )
                    # Sync promoted or active persistent ID
                    current_gid = id_mgr.local_to_global.get(tid, gid)
                    if current_gid and current_gid.startswith("P-"):
                        active_gids.add(current_gid)
                        dec['global_id'] = current_gid
                        if event == "IN":
                            dec['state'] = "NEW"
                        if det.get('crop') is not None and getattr(self, "vlm_verifier", None) and self.vlm_verifier.enabled:
                            ref_crops = self.vlm_verifier.reference_crops.get(current_gid, [])
                            if len(ref_crops) < 3:
                                bw = det['bbox'][2] - det['bbox'][0]
                                bh = det['bbox'][3] - det['bbox'][1]
                                if bw >= 20 and bh >= 45:
                                    self.vlm_verifier.store_reference_crop(current_gid, det['crop'])

                # Check doorway departures among registered people who were inside and departed through the door
                lost_gids = set(id_mgr.persons.keys()) - active_gids
                departures = self.tripwire.check_departures(active_gids, lost_gids, frame_w, frame_h, timestamp=timestamp)
                for ev, pid, ts in departures:
                    id_mgr.terminate_visit(pid, ts, is_short=False)

                # ── 3.5 Group Detection (only for moving, active people) ───
                active_persons = []
                for det in detections:
                    tid = det['track_id']
                    if tid in dec_map:
                        dec = dec_map[tid]
                        gid = dec['global_id']
                        if gid and gid.startswith("P-"):
                            p_obj = id_mgr.persons.get(gid)
                            if p_obj and not p_obj.is_departed:
                                active_persons.append({
                                    'global_id': gid,
                                    'bbox': det['bbox']
                                })

                group_assignments = self.processor.group_detector.update(active_persons)

                # ── 4. Visualize on Frame ──────────────────────────────────
                statues_count = 0
                for det in detections:
                    tid = det['track_id']
                    if tid not in dec_map:
                        continue
                    dec   = dec_map[tid]
                    gid   = dec['global_id']
                    state = dec['state']
                    sim   = dec.get('sim') or 0.0

                    # 1. Statue or mannequin: dimmed grey box
                    if gid == "STATUE":
                        statues_count += 1
                        if self.show_statues:
                            x1, y1, x2, y2 = det['bbox']
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1, cv2.LINE_AA)
                            cv2.putText(frame, "[STATUE/MANNEQUIN - IGNORED]", (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
                        continue

                    # 2. Unpromoted track or unverified occupant: outline with temporary tracking ID
                    if not (gid and gid.startswith("P-")):
                        x1, y1, x2, y2 = det['bbox']
                        fx, fy = int((x1 + x2) / 2.0), int(y2)
                        p_key = gid if (gid and gid.startswith("P-")) else tid
                        dbg_info = self.tripwire.last_eval_debug.get(p_key) or self.tripwire.last_eval_debug.get(tid)
                        d_val = dbg_info.get('dist', 0.0) if dbg_info else 0.0
                        st_val = (dbg_info.get('new_state') or state) if dbg_info else state
                        dbg_txt = f"{st_val} d:{d_val:+.0f}px"

                        if state == "UNVERIFIED_OCCUPANT":
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1, cv2.LINE_AA)
                            tag = f"Unverified Occupant [{tid}]"
                            cv2.putText(frame, tag, (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                            cv2.circle(frame, (fx, fy), 4, (180, 180, 180), -1)
                            cv2.putText(frame, dbg_txt, (x1, min(frame_h - 4, y2 + 14)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 1, cv2.LINE_AA)
                            tag = f"Approaching [{tid}]" if state == "OUTSIDE" else f"Inside [{tid}]"
                            cv2.putText(frame, tag, (x1, max(15, y1 - 6)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)
                            cv2.circle(frame, (fx, fy), 4, (0, 229, 255), -1)
                            cv2.putText(frame, dbg_txt, (x1, min(frame_h - 4, y2 + 14)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 229, 255), 1, cv2.LINE_AA)
                        continue

                    p_obj = id_mgr.persons.get(gid)
                    if p_obj and (p_obj.is_departed or getattr(p_obj, "current_visit_status", None) in ("SHORT_VISIT_IGNORED", "COMPLETED")):
                        continue

                    # 3. Permanent visitor: log tracking and draw full colored box
                    self.processor.event_logger.log_tracking(
                        timestamp, current_frame_idx, cam_id, tid, gid,
                        det['bbox'], det['conf'], sim, state
                    )

                    x1, y1, x2, y2 = det['bbox']
                    fx, fy = int((x1 + x2) / 2.0), int(y2)

                    # Apply group colors and labels
                    if gid in group_assignments:
                        grp_id = group_assignments[gid]
                        color_hash = (grp_id * 1234567) % 16777215
                        r = (color_hash & 0xFF0000) >> 16
                        g_col = (color_hash & 0x00FF00) >> 8
                        b = color_hash & 0x0000FF
                        color = (max(50, r), max(50, g_col), max(50, b))

                        if state == "REIDENTIFIED":
                            label = f"{gid} [G{grp_id}] (reid {sim:.2f})"
                        else:
                            label = f"{gid} [G{grp_id}]"
                    else:
                        if state == "REIDENTIFIED":
                            color, label = (0, 165, 255), f"{gid} (reid {sim:.2f})"
                        elif state == "NEW":
                            color, label = (0, 200, 255), f"{gid}"
                        else:
                            color, label = (0, 255, 0), gid

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame,
                                  (x1, max(0, y1 - th - 8)),
                                  (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    # Debug foot point and distance overlay
                    cv2.circle(frame, (fx, fy), 4, color, -1)
                    dbg_info = self.tripwire.last_eval_debug.get(person_key) or self.tripwire.last_eval_debug.get(tid)
                    if dbg_info:
                        d_val = dbg_info.get('dist', 0.0)
                        st_val = dbg_info.get('new_state') or state
                        cv2.putText(frame, f"d:{d_val:+.0f}px [{st_val}]", (x1, min(frame_h - 4, y2 + 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

                # ── 5. Stats counts (Strictly excluding statues, departed & short visits) ──
                n_active = sum(1 for p in id_mgr.persons.values()
                               if p.status in ("ACTIVE", "REIDENTIFIED") and not p.is_departed)
                n_unique = sum(1 for p in id_mgr.persons.values()
                               if getattr(p, "current_visit_status", None) != "SHORT_VISIT_IGNORED")
                n_lost   = sum(1 for p in id_mgr.persons.values()
                               if p.status == "TEMPORARILY_LOST" and not p.is_departed)

                total_statues = statues_count

                in_cnt = self.tripwire.entry_count
                out_cnt = self.tripwire.exit_count
                groups_cnt = len(self.processor.group_detector.group_records)

                # ── 6. Send to Qt ──────────────────────────────────────────
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h_f, w_f, ch = rgb.shape
                qimg = QImage(rgb.data, w_f, h_f, ch * w_f,
                              QImage.Format_RGB888).copy()
                self.frame_ready.emit(cam_id, qimg)
                self.stats_updated.emit(n_active, n_unique, n_lost, in_cnt, out_cnt, total_statues, groups_cnt)

            if all_frames_empty:
                break

            pct = int((frame_idx / total_frames) * 100)
            self.progress_updated.emit(pct)

            frame_idx += 1

        self.processor.event_logger.save_global_summary(id_mgr.persons)
        self.finished.emit()

    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Retail CCTV Analytics – Tracking Area & Gate Entry/Exit")
        self.resize(1360, 860)
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")

        self.cameras          = []
        self.processor_thread = None
        self._thread          = None
        self.preview_frame    = None

        self.init_ui()

    # ── UI setup ─────────────────────────────────────────────────────────────
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ── Left panel (Controls & Statistics) ────────────────────────────
        left = QVBoxLayout()
        left.setAlignment(Qt.AlignTop)
        left.setSpacing(8)

        # 1. Video Selection
        btn_add = QPushButton("➕  Select Video File (Instant Preview)")
        btn_add.setMinimumHeight(44)
        btn_add.setStyleSheet(self._btn_style("#2979ff"))
        btn_add.clicked.connect(self.add_video)
        left.addWidget(btn_add)

        self.cam_table = QTableWidget(0, 2)
        self.cam_table.setHorizontalHeaderLabels(["Camera ID", "File Path"])
        self.cam_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.cam_table.setMaximumHeight(90)
        self.cam_table.setStyleSheet("background:#252525; color:#fff; gridline-color:#3a3a3a; border-radius:4px;")
        left.addWidget(self.cam_table)

        # 2. Controls
        ctrl_box = QGroupBox("Processing Controls")
        ctrl_box.setStyleSheet("QGroupBox { font-weight: bold; color: #aaa; border: 1px solid #333; "
                              "border-radius: 6px; margin-top: 6px; padding-top: 10px; }")
        ctrl_layout = QVBoxLayout(ctrl_box)
        ctrl_layout.setSpacing(6)

        self.btn_start = QPushButton("▶  Start Processing")
        self.btn_start.setMinimumHeight(42)
        self.btn_start.setStyleSheet(self._btn_style("#4caf50"))
        self.btn_start.clicked.connect(self.start_processing)
        ctrl_layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("⏸  Pause / Resume")
        self.btn_pause.setMinimumHeight(38)
        self.btn_pause.setStyleSheet(self._btn_style("#ff9800"))
        self.btn_pause.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        ctrl_layout.addWidget(self.btn_pause)

        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setStyleSheet(self._btn_style("#f44336"))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)
        ctrl_layout.addWidget(self.btn_stop)

        self.btn_refresh = QPushButton("🔄  Reset Video & All Settings")
        self.btn_refresh.setMinimumHeight(36)
        self.btn_refresh.setStyleSheet(self._btn_style("#757575"))
        self.btn_refresh.clicked.connect(self.refresh_ui)
        ctrl_layout.addWidget(self.btn_refresh)

        # Process Every Nth Frame (Frame Skip)
        skip_row = QHBoxLayout()
        lbl_skip = QLabel("Process Every Nth Frame:")
        lbl_skip.setStyleSheet("font-size: 12px; color: #bbb; font-weight: bold;")
        self.spin_frame_skip = QSpinBox()
        self.spin_frame_skip.setRange(1, 30)
        self.spin_frame_skip.setValue(1)
        self.spin_frame_skip.setToolTip("1 = Process every frame (default)\n2 = Process every second frame\n5 = Process every fifth frame")
        self.spin_frame_skip.setStyleSheet("background:#252525; color:#fff; font-size:13px; padding:3px; border:1px solid #444; border-radius:4px;")
        skip_row.addWidget(lbl_skip)
        skip_row.addWidget(self.spin_frame_skip)
        ctrl_layout.addLayout(skip_row)

        self.lbl_skip_warn = QLabel("⚠️ Higher frame skipping increases processing speed but may cause missed detections, identity switches, and inaccurate gate counts.")
        self.lbl_skip_warn.setWordWrap(True)
        self.lbl_skip_warn.setStyleSheet("font-size: 10px; color: #ffb74d; margin-top: 2px;")
        ctrl_layout.addWidget(self.lbl_skip_warn)

        # Re-ID Match Threshold (%)
        reid_row = QHBoxLayout()
        lbl_reid = QLabel("Re-ID Match Threshold:")
        lbl_reid.setStyleSheet("font-size: 12px; color: #bbb; font-weight: bold;")
        self.spin_reid_thresh = QSpinBox()
        self.spin_reid_thresh.setRange(10, 99)
        self.spin_reid_thresh.setValue(int(getattr(Config, "REID_MATCH_THRESHOLD", 0.75) * 100))
        self.spin_reid_thresh.setSuffix("%")
        self.spin_reid_thresh.setToolTip("Minimum appearance similarity required to match tracks (default: 75%).")
        self.spin_reid_thresh.setStyleSheet("background:#252525; color:#fff; font-size:13px; padding:3px; border:1px solid #444; border-radius:4px;")
        self.spin_reid_thresh.valueChanged.connect(self.on_reid_threshold_changed)
        reid_row.addWidget(lbl_reid)
        reid_row.addWidget(self.spin_reid_thresh)
        ctrl_layout.addLayout(reid_row)

        # Active Visitor Recovery Threshold (%)
        active_reid_row = QHBoxLayout()
        lbl_active_reid = QLabel("Active Visitor Recovery:")
        lbl_active_reid.setStyleSheet("font-size: 12px; color: #bbb; font-weight: bold;")
        self.spin_active_reid_thresh = QSpinBox()
        self.spin_active_reid_thresh.setRange(10, 99)
        self.spin_active_reid_thresh.setValue(int(getattr(Config, "ACTIVE_VISITOR_REID_THRESHOLD", 0.75) * 100))
        self.spin_active_reid_thresh.setSuffix("%")
        self.spin_active_reid_thresh.setToolTip("Minimum appearance similarity required to recover an active visitor after occlusion/disappearance (default: 75%).")
        self.spin_active_reid_thresh.setStyleSheet("background:#252525; color:#fff; font-size:13px; padding:3px; border:1px solid #444; border-radius:4px;")
        self.spin_active_reid_thresh.valueChanged.connect(self.on_active_reid_threshold_changed)
        active_reid_row.addWidget(lbl_active_reid)
        active_reid_row.addWidget(self.spin_active_reid_thresh)
        # Local VLM Verifier (Ollama)
        self.chk_vlm = QCheckBox("Enable Local VLM Verifier (Ollama Qwen 3.5)")
        self.chk_vlm.setChecked(getattr(Config, "VLM_ENABLED", True))
        self.chk_vlm.setStyleSheet("font-size: 12px; color: #00e5ff; font-weight: bold; margin-top: 4px;")
        self.chk_vlm.setToolTip("Uses local Ollama Qwen 3.5 vision model to verify ambiguous person re-identifications.")
        self.chk_vlm.toggled.connect(self.on_vlm_toggled)
        ctrl_layout.addWidget(self.chk_vlm)

        left.addWidget(ctrl_box)

        # 3. Area & Gate Setup Tools
        setup_box = QGroupBox("Tracking Area & Entry Gate Tools")
        setup_box.setStyleSheet("QGroupBox { font-weight: bold; color: #ffb300; border: 1px solid #443a00; "
                               "border-radius: 6px; margin-top: 6px; padding-top: 10px; }")
        setup_layout = QVBoxLayout(setup_box)
        setup_layout.setSpacing(6)

        hint = QLabel("💡 Tip: Click & drag the Cyan boundary lines to limit tracking area. Drag the Amber line & dots to place the Entry Gate!")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; color: #bbb; line-height: 1.3;")
        setup_layout.addWidget(hint)

        btn_flip_gate = QPushButton("⇄  Flip Gate IN / OUT Direction")
        btn_flip_gate.setMinimumHeight(34)
        btn_flip_gate.setStyleSheet(self._btn_style("#ff8f00"))
        btn_flip_gate.clicked.connect(self.flip_gate_direction)
        setup_layout.addWidget(btn_flip_gate)

        btn_reset_lines = QPushButton("🎯  Reset Lines to Default")
        btn_reset_lines.setMinimumHeight(34)
        btn_reset_lines.setStyleSheet(self._btn_style("#424242"))
        btn_reset_lines.clicked.connect(self.reset_lines)
        setup_layout.addWidget(btn_reset_lines)

        self.chk_show_guides = QCheckBox("Show Guides & Lines during playback")
        self.chk_show_guides.setChecked(True)
        self.chk_show_guides.setStyleSheet("font-size: 12px; color: #ddd;")
        self.chk_show_guides.toggled.connect(self.toggle_show_guides)
        setup_layout.addWidget(self.chk_show_guides)

        left.addWidget(setup_box)

        # 4. Live Analytics & Counts Panel
        stats_box = QGroupBox("Live Store Analytics")
        stats_box.setStyleSheet("QGroupBox { font-weight: bold; color: #00e5ff; border: 1px solid #004d40; "
                                "border-radius: 6px; margin-top: 6px; padding-top: 10px; }")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(6)

        def make_stat_row(title, color="#00e5ff"):
            lbl = QLabel(title)
            lbl.setStyleSheet(f"font-size: 14px; font-weight: bold; color: {color}; padding: 2px;")
            return lbl

        self.lbl_in      = make_stat_row("🚪 Gate Entries (IN):  0", "#00e676")
        self.lbl_out     = make_stat_row("🚪 Gate Exits (OUT):  0", "#ff5252")
        self.lbl_inside  = make_stat_row("🏬 Currently Inside:  0", "#ffeb3b")
        self.lbl_groups  = make_stat_row("👥 Confirmed Groups:  0", "#e040fb")
        self.lbl_active  = make_stat_row("👤 Active Customers:  0", "#00e5ff")
        self.lbl_unique  = make_stat_row("👥 Total Unique Visitors:  0", "#40c4ff")
        self.lbl_lost    = make_stat_row("⏳ Temporarily Lost:  0", "#b0bec5")
        self.lbl_statues = make_stat_row("🗿 Statues Ignored:  0", "#9e9e9e")

        for l in [self.lbl_in, self.lbl_out, self.lbl_inside, self.lbl_groups,
                  self.lbl_active, self.lbl_unique, self.lbl_lost, self.lbl_statues]:
            stats_layout.addWidget(l)

        left.addWidget(stats_box)
        left.addStretch()

        # ── Right panel (Interactive Video Canvas & Timeline) ─────────────
        self.right_panel = QVBoxLayout()
        self.right_panel.setSpacing(6)

        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(100)
        self.timeline.setStyleSheet(
            "QSlider::groove:horizontal { height: 6px; background: #333; border-radius: 3px; }"
            "QSlider::sub-page:horizontal { background: #2979ff; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #fff; width: 14px; margin-top: -4px; margin-bottom: -4px; border-radius: 7px; }"
        )
        self.right_panel.addWidget(self.timeline)

        # Interactive Video Canvas Widget
        self.canvas = InteractiveVideoCanvas(self)
        self.canvas.gate_line_changed.connect(self.on_gate_line_changed)
        self.canvas.roi_changed.connect(self.on_roi_changed)
        self.right_panel.addWidget(self.canvas, 1)

        root.addLayout(left, 1)
        root.addLayout(self.right_panel, 3)

    @staticmethod
    def _btn_style(bg):
        return (f"background-color: {bg}; color: white; font-size: 13px;"
                " font-weight: bold; border-radius: 6px; padding: 5px;")

    # ── Slots & Actions ───────────────────────────────────────────────────────
    def on_gate_line_changed(self, x1, y1, x2, y2):
        if self.processor_thread:
            self.processor_thread.update_gate_geometry(
                (x1, y1), (x2, y2), self.canvas.gate_direction
            )

    def on_roi_changed(self, l, t, r, b):
        if self.processor_thread:
            self.processor_thread.update_roi(l, t, r, b)

    def on_reid_threshold_changed(self, val):
        Config.REID_MATCH_THRESHOLD = val / 100.0

    def on_active_reid_threshold_changed(self, val):
        Config.ACTIVE_VISITOR_REID_THRESHOLD = val / 100.0

    def on_vlm_toggled(self, checked):
        Config.VLM_ENABLED = checked
        if self.processor_thread:
            self.processor_thread.set_vlm_enabled(checked)

    def add_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.MP4 *.MOV *.AVI)")
        if path:
            cam_id = f"camera_{len(self.cameras) + 1:02d}"
            self.cameras = [(cam_id, path)]  # Primary camera view

            self.cam_table.setRowCount(0)
            self.cam_table.insertRow(0)
            self.cam_table.setItem(0, 0, QTableWidgetItem(cam_id))
            self.cam_table.setItem(0, 1, QTableWidgetItem(path))

            # ── INSTANT PREVIEW: Load frame 0 immediately upon selection ──
            cap = cv2.VideoCapture(path)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None:
                self.preview_frame = frame
                self.canvas.set_frame(frame)
            else:
                QMessageBox.warning(self, "Preview Error", "Could not read preview frame from video.")

    def flip_gate_direction(self):
        self.canvas.flip_gate_direction()
        if self.processor_thread:
            p1, p2 = self.canvas.gate_p1, self.canvas.gate_p2
            self.processor_thread.update_gate_geometry(p1, p2, self.canvas.gate_direction)

    def reset_lines(self):
        self.canvas.reset_geometry()

    def toggle_show_guides(self, checked: bool):
        self.canvas.show_guides_during_play = checked
        self.canvas.update()

    def start_processing(self):
        if not self.cameras:
            QMessageBox.warning(self, "No Video", "Please select a video file first.")
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_pause.setEnabled(True)
        self.spin_frame_skip.setEnabled(False)

        # Sync user-selected Re-ID similarity thresholds
        Config.REID_MATCH_THRESHOLD = self.spin_reid_thresh.value() / 100.0
        Config.ACTIVE_VISITOR_REID_THRESHOLD = self.spin_active_reid_thresh.value() / 100.0

        # Grab current interactive ROI and Gate Line from canvas
        roi_norm = (self.canvas.roi_left, self.canvas.roi_top,
                    self.canvas.roi_right, self.canvas.roi_bottom)
        gate_norm = (self.canvas.gate_p1, self.canvas.gate_p2)
        gate_dir = self.canvas.gate_direction

        self.processor_thread = VideoProcessorThread(
            self.cameras,
            roi_norm=roi_norm,
            gate_norm=gate_norm,
            gate_dir=gate_dir,
            show_statues=True,
            frame_skip=self.spin_frame_skip.value()
        )

        self.processor_thread.frame_ready.connect(self.update_frame)
        self.processor_thread.stats_updated.connect(self.update_stats)
        self.processor_thread.progress_updated.connect(self.update_progress)
        self.processor_thread.finished.connect(self.processing_finished)

        self._thread = threading.Thread(target=self.processor_thread.run, daemon=True)
        self._thread.start()

    def update_frame(self, cam_id, qimg):
        self.canvas.set_frame(qimg)

    def update_stats(self, active, unique, lost, in_count, out_count, statues, groups=0):
        self.lbl_in.setText(f"🚪 Gate Entries (IN):  {in_count}")
        self.lbl_out.setText(f"🚪 Gate Exits (OUT):  {out_count}")
        inside = max(0, in_count - out_count)
        self.lbl_inside.setText(f"🏬 Currently Inside:  {inside}")
        self.lbl_groups.setText(f"👥 Confirmed Groups:  {groups}")
        self.lbl_active.setText(f"👤 Active Customers:  {active}")
        self.lbl_unique.setText(f"👥 Total Unique Visitors:  {unique}")
        self.lbl_lost.setText(f"⏳ Temporarily Lost:  {lost}")
        self.lbl_statues.setText(f"🗿 Statues Ignored:  {statues}")

        self.canvas.set_counts(in_count, out_count, statues, groups)

    def update_progress(self, pct):
        self.timeline.setValue(pct)

    def stop_processing(self):
        if self.processor_thread:
            self.processor_thread.stop()
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("⏸  Pause / Resume")
        self.btn_start.setEnabled(True)
        self.spin_frame_skip.setEnabled(True)

    def toggle_pause(self):
        if self.processor_thread:
            self.processor_thread.paused = not self.processor_thread.paused
            if self.processor_thread.paused:
                self.btn_pause.setText("▶  Resume")
            else:
                self.btn_pause.setText("⏸  Pause / Resume")

    def refresh_ui(self):
        self.stop_processing()
        self.cameras = []
        self.cam_table.setRowCount(0)
        self.timeline.setValue(0)
        self.canvas.reset_geometry()
        self.canvas._pixmap = None
        self.canvas.set_counts(0, 0, 0, 0)
        self.update_stats(0, 0, 0, 0, 0, 0, 0)
        self.canvas.update()
        self.btn_start.setEnabled(True)
        self.spin_frame_skip.setEnabled(True)
        self.spin_reid_thresh.setValue(75)
        self.spin_active_reid_thresh.setValue(75)

    def processing_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.spin_frame_skip.setEnabled(True)
        QMessageBox.information(
            self,
            "Done",
            f"Processing finished!\n\n"
            f"Total Entries (IN): {self.canvas.live_in_count}\n"
            f"Total Exits (OUT): {self.canvas.live_out_count}\n"
            f"Final Inside Occupancy: {max(0, self.canvas.live_in_count - self.canvas.live_out_count)}\n"
            f"Confirmed Groups Formed: {self.canvas.live_group_count}\n"
            f"Statues / Mannequins Ignored: {self.canvas.live_statues_ignored}\n\n"
            f"Summary saved to outputs/ directory."
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
