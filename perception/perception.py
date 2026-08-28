

import os, sys, re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from larp_agent import Scene, object_class, affordances_of
except Exception:
    def object_class(oid): return re.sub(r"_\d+$", "", oid)
    _AFF = {"plate": ["surface", "container"], "bowl": ["container"],
            "cup": ["container"], "fridge": ["openable", "container"],
            "table": ["surface"]}
    def affordances_of(cls): return _AFF.get(cls, ["graspable"])
    class Scene:
        def __init__(self, objects, held=None, open_objects=None, on=None, inside=None):
            self.objects = objects; self.held = held
            self.open_objects = open_objects or []
            self.on = on or []; self.inside = inside or []


@dataclass
class Detection:
    cls: str
    inst_id: str
    box: Tuple[float, float, float, float]
    conf: float
    @property
    def center(self):
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    @property
    def area(self):
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def detect(image_path: str, classes: List[str],
           weights: str = "yoloe-v8s-seg.pt",
           conf: float = 0.45, iou: float = 0.5,
           save: bool = True, out_project: str = "perception_out") -> List[Detection]:
    
    from ultralytics import YOLOE
    model = YOLOE(weights)
    model.set_classes(classes, model.get_text_pe(classes))
    results = model.predict(image_path, conf=conf, iou=iou,
                            save=save, project=out_project, name="scene", exist_ok=True)
    r = results[0]
    names = r.names
    dets: List[Detection] = []
    counts: Dict[str, int] = {}
    boxes = sorted(r.boxes, key=lambda b: float(b.conf), reverse=True)
    for b in boxes:
        cls = names[int(b.cls)]
        counts[cls] = counts.get(cls, 0) + 1
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        dets.append(Detection(cls, f"{cls}_{counts[cls]}", (x1, y1, x2, y2), float(b.conf)))
    return dets


def _center_inside(a: Detection, b: Detection) -> bool:
    ax, ay = a.center
    bx1, by1, bx2, by2 = b.box
    return bx1 <= ax <= bx2 and by1 <= ay <= by2

def _overlap_ratio(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / a.area if a.area > 0 else 0.0


SUPPORT_CLASSES = {"plate", "table", "bowl", "tray", "cuttingboard"}
CONTAINER_CLASSES = {"fridge", "drawer", "cabinet", "bowl", "cup"}

def infer_relations(dets: List[Detection], overlap_thresh: float = 0.5):
    
    on, inside = [], []
    for a in dets:
        best_support: Optional[Detection] = None
        trapped = False
        for b in dets:
            if a is b or not _center_inside(a, b) or a.area >= b.area:
                continue
            bcls = object_class(b.inst_id)
            if bcls in CONTAINER_CLASSES and _overlap_ratio(a, b) >= overlap_thresh:
                inside.append((a.inst_id, b.inst_id)); trapped = True; break
            if bcls in SUPPORT_CLASSES or "surface" in affordances_of(bcls):
                if best_support is None or b.area < best_support.area:
                    best_support = b
        if not trapped and best_support is not None:
            on.append((a.inst_id, best_support.inst_id))
    return on, inside


def build_scene(image_path: str, classes: List[str],
                weights: str = "yoloe-v8s-seg.pt",
                conf: float = 0.45, iou: float = 0.5,
                assume_table: bool = True):
    
    dets = detect(image_path, classes, weights=weights, conf=conf, iou=iou)
    has_table = any(object_class(d.inst_id) == "table" for d in dets)
    object_ids = [d.inst_id for d in dets]
    if assume_table and not has_table:
        object_ids.append("table_1")
    on, inside = infer_relations(dets)
    supported = {a for a, _ in on} | {a for a, _ in inside}
    if assume_table:
        for d in dets:
            if d.inst_id not in supported and object_class(d.inst_id) != "table":
                on.append((d.inst_id, "table_1"))
    scene = Scene(objects=object_ids, held=None, open_objects=[], on=on, inside=inside)
    return scene, dets


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", default="yoloe-v8s-seg.pt")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--classes", nargs="+",
                    default=["knife", "cup", "plate", "bread", "Sliced tomato",
                             "fork", "ground beef", "lettuce"])
    args = ap.parse_args()
    scene, dets = build_scene(args.image, args.classes,
                              weights=args.weights, conf=args.conf, iou=args.iou)
    print("\n=== DETECTIONS ===")
    for d in dets:
        cx, cy = d.center
        print(f"  {d.inst_id:<12} conf={d.conf:.2f}  center=({cx:.0f},{cy:.0f})")
    print("\n=== SCENE GRAPH ===")
    print("objects:", scene.objects)
    print("on     :", scene.on)
    print("inside :", scene.inside)
    print("\n(Annotated image saved under perception_out/scene/)")
    print("Pass `scene` straight to LARPAgent.plan(command, scene).")
