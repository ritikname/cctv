class CameraManager:
    def __init__(self):
        self.cameras = {}

    def add_camera(self, camera_id, name, connected_cameras=None, stream_url=None):
        if connected_cameras is None:
            connected_cameras = []
        self.cameras[camera_id] = {
            'name': name,
            'connected_cameras': connected_cameras,
            'stream_url': stream_url
        }

    def get_camera(self, camera_id):
        return self.cameras.get(camera_id)

    def is_connected(self, cam1, cam2):
        if cam1 not in self.cameras:
            return False
        # Treat as bidirectionally connected if in either list
        cam1_connected = cam2 in self.cameras[cam1]['connected_cameras']
        cam2_connected = cam1 in self.cameras.get(cam2, {}).get('connected_cameras', [])
        return cam1_connected or cam2_connected
        
    def check_transition_plausible(self, cam1, cam2, time_gap, max_gap):
        if cam1 == cam2:
            return True
            
        if time_gap > max_gap:
            return False
            
        # Optional: check if transition between cam1 and cam2 is physically possible
        # within time_gap. For simplicity, we just check if they are logically connected.
        if not self.is_connected(cam1, cam2):
            # If not explicitly connected, maybe we reject or penalize
            # For now, allow it but we could return False to strictly enforce topology
            pass
            
        return True
