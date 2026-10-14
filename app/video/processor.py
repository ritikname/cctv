import cv2
import time
from app.tracking.local_tracker import LocalTracker
from app.reid.model import ReIDModel
from app.identity.global_identity_manager import GlobalIdentityManager
from app.cameras.camera_manager import CameraManager
from app.events.event_logger import EventLogger
from app.analytics.group_detector import GroupDetector
from app.config import Config
import numpy as np

class MultiCameraProcessor:
    def __init__(self):
        self.camera_manager = CameraManager()
        self.reid_model = ReIDModel()
        self.identity_manager = GlobalIdentityManager(self.camera_manager)
        self.group_detector = GroupDetector()
        self.event_logger = EventLogger(Config.OUTPUT_DIR)
        
        self.trackers = {}
        self.video_caps = {}
        
    def add_camera_stream(self, camera_id, video_path):
        self.camera_manager.add_camera(camera_id, camera_id)
        self.trackers[camera_id] = LocalTracker(camera_id)
        self.video_caps[camera_id] = cv2.VideoCapture(video_path)
        
    def run(self, frame_skip: int = 1):
        frame_skip = max(1, int(frame_skip))
        frame_idx = 0
        
        while True:
            all_frames_empty = True
            
            for cam_id, cap in self.video_caps.items():
                ret, frame = cap.read()
                if not ret:
                    continue
                
                all_frames_empty = False
                current_frame_idx = frame_idx
                
                if current_frame_idx % frame_skip != 0:
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                timestamp = current_frame_idx / fps # accurate video timeline
                
                # 1. Local Tracking
                detections = self.trackers[cam_id].process_frame(frame)
                
                # 2. Extract Re-ID Features Selectively
                features_dict = {}
                crops_to_extract = []
                ids_to_extract = []
                
                for d in detections:
                    loc_id = d['track_id']
                    needs_feature = False
                    
                    if loc_id not in self.identity_manager.local_to_global:
                        needs_feature = True
                    else:
                        gid = self.identity_manager.local_to_global[loc_id]
                        person = self.identity_manager.persons[gid]
                        if person.gallery.size() < 5 or frame_idx % 15 == 0:
                            needs_feature = True
                            
                    if needs_feature:
                        crops_to_extract.append(d['crop'])
                        ids_to_extract.append(loc_id)
                        
                if crops_to_extract:
                    extracted_feats = self.reid_model.extract_features(crops_to_extract)
                    for i, loc_id in enumerate(ids_to_extract):
                        features_dict[loc_id] = extracted_feats[i]
                        
                # 3. Global Identity Management
                decisions = self.identity_manager.process_detections(detections, features_dict, timestamp, frame_idx)
                
                # 4. Logging & Visualization
                dec_map = {d['track_id']: d for d in decisions}
                
                # --- Group Detection ---
                active_persons = []
                for det in detections:
                    loc_id = det['track_id']
                    if loc_id in dec_map and dec_map[loc_id]['state'] != "PENDING":
                        active_persons.append({
                            'global_id': dec_map[loc_id]['global_id'],
                            'bbox': det['bbox']
                        })
                
                group_assignments = self.group_detector.update(active_persons)
                
                for det in detections:
                    loc_id = det['track_id']
                    if loc_id not in dec_map:
                        continue
                        
                    dec = dec_map[loc_id]
                    gid = dec['global_id']
                    state = dec['state']
                    sim = dec.get('sim', 0.0)
                    
                    if state == "PENDING":
                        continue
                        
                    self.event_logger.log_tracking(
                        timestamp, frame_idx, cam_id, loc_id, gid, 
                        det['bbox'], det['conf'], sim, state
                    )
                    
                    if state == "NEW":
                        self.event_logger.log_event(f"NEW PERSON\n{gid}\nCamera: {cam_id}")
                    elif state == "REIDENTIFIED":
                        self.event_logger.log_event(f"RE-ID SUCCESS\n{loc_id} -> {gid}\nSimilarity: {sim:.2f}")
                        
                    x1, y1, x2, y2 = det['bbox']
                    
                    # --- Color and Label Logic ---
                    if gid in group_assignments:
                        grp_id = group_assignments[gid]
                        # Use deterministic hash for distinct colors
                        color_hash = (grp_id * 1234567) % 16777215
                        r = (color_hash & 0xFF0000) >> 16
                        g_col = (color_hash & 0x00FF00) >> 8
                        b = color_hash & 0x0000FF
                        color = (max(50, r), max(50, g_col), max(50, b))
                        label = f"{gid} [Group {grp_id}]"
                    else:
                        color = (0, 255, 0) if state == "ACTIVE" else (0, 165, 255)
                        label = f"{gid}"
                        
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                                
                # Debug display (excluding statues/mannequins)
                active = len([p for p in self.identity_manager.persons.values() if p.status == "ACTIVE" and not p.is_static])
                unique = len([p for p in self.identity_manager.persons.values() if not p.is_static])
                
                cv2.putText(frame, f"Active: {active} | Unique: {unique}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                            
                cv2.imshow(f"Camera: {cam_id}", frame)
                
            if all_frames_empty:
                break
                
            frame_idx += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        # Cleanup
        for cap in self.video_caps.values():
            cap.release()
        cv2.destroyAllWindows()
        
        # Save summary
        self.event_logger.save_global_summary(self.identity_manager.persons)
        
        # Dump Re-ID audit log for diagnosability
        import os
        reid_log_path = os.path.join(Config.OUTPUT_DIR, "reid_audit_log.csv")
        self.identity_manager.dump_reid_log(reid_log_path)
        print(f"Processing finished. Summary saved. Re-ID log: {reid_log_path}")
