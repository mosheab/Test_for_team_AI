from typing import Tuple, Dict, Any
import cv2, torch, numpy as np
import torchvision.transforms as T
from torchvision import models

_resnet = None
_idx_to_label = None

def _load_model():
    global _resnet, _idx_to_label
    if _resnet is not None: return
    try:
        weights = models.ResNet50_Weights.DEFAULT
        _resnet = models.resnet50(weights=weights)
        _idx_to_label = weights.meta.get("categories", None)
    except Exception:
        _resnet = models.resnet50(pretrained=True)
        _idx_to_label = None
    _resnet.eval()

def _labels_for(idx_list):
    if _idx_to_label:
        return [_idx_to_label[i] for i in idx_list]
    return [f"class_{i}" for i in idx_list]

def keyframe_objects(video_path: str, scene: Tuple[float, float]) -> Dict[str, Any]:
    _load_model()
    start, end = scene
    mid = (start + end) / 2.0

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = int(mid * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return {"objects": [], "motion": 0.0}

    # motion estimate around mid
    motion = 0.0
    cap = cv2.VideoCapture(video_path)
    start_f = max(0, frame_no - 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    prev = None; cnt = 0
    for _ in range(10):
        ok, f = cap.read()
        if not ok: break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            motion += float(np.mean(cv2.absdiff(gray, prev))); cnt += 1
        prev = gray
    cap.release()
    motion = motion / max(1, cnt)

    transform = T.Compose([
        T.ToPILImage(), T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    x = transform(frame).unsqueeze(0)
    with torch.no_grad():
        logits = _resnet(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    top_idx = probs.argsort()[-5:][::-1].tolist()
    labels = _labels_for(top_idx)
    return {"objects": labels, "motion": motion}
