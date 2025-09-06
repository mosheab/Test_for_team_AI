from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector

def detect_scenes(video_path: str, threshold: int = 5) -> list[tuple[float, float]]:
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]
