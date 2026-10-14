import os
import json
import csv
from datetime import datetime

class EventLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.output_dir, f"events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.csv_file = os.path.join(self.output_dir, f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Init CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame', 'camera_id', 'track_id', 
                'global_person_id', 'x1', 'y1', 'x2', 'y2', 
                'detection_confidence', 'reid_similarity', 'state'
            ])

    def log_event(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_msg = f"{timestamp}\n{message}\n"
        print(full_msg)
        with open(self.log_file, 'a') as f:
            f.write(full_msg + "\n")

    def log_tracking(self, timestamp, frame, camera_id, track_id, gid, bbox, conf, reid_sim, state):
        x1, y1, x2, y2 = bbox
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, frame, camera_id, track_id, 
                gid, x1, y1, x2, y2, 
                conf, reid_sim, state
            ])

    def save_global_summary(self, persons):
        summary = []
        for gid, p in persons.items():
            summary.append({
                "global_person_id": p.global_person_id,
                "first_seen": p.first_seen,
                "last_seen": p.last_seen,
                "cameras_seen": p.camera_history,
                "total_visible_frames": p.total_visible_frames,
                "reidentification_count": len(p.reidentification_events),
                "times_lost": p.times_lost
            })
            
        summary_file = os.path.join(self.output_dir, f"visitor_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
