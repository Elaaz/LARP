

import os, sys, re, json, time, copy, argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "perception", "strategic", "grasping", "robot", "model", "domain"]:
    p = os.path.join(_ROOT, sub)
    if os.path.isdir(p) or sub == "":
        sys.path.insert(0, p)

import numpy as np
import cv2
import delta_manager.camera as Camera
from delta_manager.delta_manager import DeltaManager


@dataclass
class Config:
    live: bool = False
    test_image: str = None
    auto_confirm: bool = False                                                   
    above_offset: float = 6.0
                                                                               
    roi_margin_cm: float = 2.0
                                                                                
                                                                              
                                                                               
    px_per_cm: float = 37.0
    z_fom: float = 0.0
    gripper_type: str = "New_Hand"
    z_obj: Dict[str, float] = field(default_factory=lambda: {
        "knife": 1.9, "cup": 8.0, "plate_2":4.0, "bread_1":0.00, "bread_2":4.0, "plate_1":0.05, "plate_3": 0.15,
        "tomato": 4.0, "apple": 7.0, "bowl": 6.0, "ground beef":4.25, "fork": 1.5, "lettuce": 4.0})
    default_z_obj: float = 2.0
    clear_location_xyz: Tuple[float,float,float] = (15.0, 15.0, -59.0)                                   
    release_clearance: float = 0.5                                                                  
                                                                               
                                                                           
                                                     
    classes: List[str] = field(default_factory=lambda: ["plate","bread","sliced tomato","ground beef","lettuce"])
    yolo_conf: float = 0.05                                                      
    yolo_iou: float = 0.5
    require_real_detection: bool = True                                          
    cam_index: int = 1
    cam_w: int = 4000
    cam_h: int = 4000
    capturing_coord: Tuple[float,float,float] = (0.0, 0.0, -37.0)
    sensetask_objects: Tuple[str,...] = ("knife", "cup")                         
    sensetask_repo: str = "grasping/SenseTask"
    sensetask_models_path: str = "grasping/SenseTask/models"                                      
    log_path: str = "larp_run_log.json"

ACTION_TO_TASK = {"slice":"Tool Usage","pour":"Pouring",
                  "open":"Opening_Closing","close":"Opening_Closing","pick":"Stabilizing"}
GRASP_ACTIONS = {"pick","slice","open","close","pour"}
def obj_class(oid): return re.sub(r"_\d+$", "", oid)

@dataclass
class Det:
    inst_id: str; cls: str; center: Tuple[float,float]
    box: Tuple[float,float,float,float]; conf: float


class LARPProduction:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log: List[Dict] = []
        self.held: Optional[str] = None
        self.last_pixel: Dict[str, Tuple[float,float]] = {}
        self._stack_height = {}                                                

        self.Delta = DeltaManager(debug_mode=(not cfg.live))
        if cfg.live:
            self.Delta.connect_gripper()

        self.cap = None
        if cfg.test_image is None and cfg.live:
            self.cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)

        self._yolo = None
        self._sensetask = None
                                            
        st_repo = os.path.join(_ROOT, cfg.sensetask_repo)
        if os.path.isdir(st_repo):
            sys.path.insert(0, st_repo)

    def _load_yolo(self):
        if self._yolo is None:
            from ultralytics import YOLOE
            self._yolo = YOLOE("yoloe-v8s-seg.pt")
            self._yolo.set_classes(self.cfg.classes, self._yolo.get_text_pe(self.cfg.classes))

                          
    def capture(self):
        if self.cfg.test_image is not None:
            frame = cv2.imread(self.cfg.test_image)
            if frame is None: raise RuntimeError(f"cannot read {self.cfg.test_image}")
            return Camera.undistort(frame)
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok: raise RuntimeError("camera read failed")
            return Camera.undistort(frame)
        return None
    


                         
    def detect(self, frame):
        if frame is None:
            if self.cfg.require_real_detection:
                raise RuntimeError(
                    "No image/camera frame, but require_real_detection=True. "
                    "Run with --image PATH or --live; refusing to use stub data "
                    "(this is what made a fake 'apple' appear before).")
            print("    [WARNING] no frame -> STUB detections (fake data!)")
            return self._stub_detections()
        self._load_yolo()
        res = self._yolo.predict([frame], conf=self.cfg.yolo_conf,
                                 iou=self.cfg.yolo_iou, verbose=False)[0]
        names = res.names
        dets, counts = [], {}
        for b in sorted(res.boxes, key=lambda b: float(b.conf), reverse=True):
            cls = names[int(b.cls)]
            counts[cls] = counts.get(cls, 0) + 1
            x1,y1,x2,y2 = [float(v) for v in b.xyxy[0].tolist()]
            dets.append(Det(f"{cls}_{counts[cls]}", cls, ((x1+x2)/2,(y1+y2)/2),
                            (x1,y1,x2,y2), float(b.conf)))
        if not dets:
            print("    [WARNING] YOLO detected nothing. Lower --conf or check the image.")
        return dets

    def _stub_detections(self):
        removed = getattr(self, "_stub_removed", set())
        base = [Det("plate_1","plate",(300,200),(260,160,340,240),0.95),
                Det("apple_1","apple",(300,200),(285,185,315,215),0.88),
                Det("knife_1","knife",(480,300),(440,290,520,310),0.90),
                Det("tomato_1","tomato",(160,350),(140,330,180,370),0.86)]
        out = [d for d in base if d.inst_id not in removed]
        if self.held: out = [d for d in out if d.inst_id != self.held]
        return out

    def build_scene(self, dets):
        try:
            from larp_agent import Scene
        except Exception:
            class Scene:
                def __init__(s, objects, held=None, open_objects=None, on=None, inside=None):
                    s.objects=objects; s.held=held
                    s.open_objects=open_objects or []; s.on=on or []; s.inside=inside or []
        ids = [d.inst_id for d in dets]
        if not any(obj_class(i)=="table" for i in ids): ids.append("table_1")
        on = []
        for a in dets:
            for b in dets:
                if a is b: continue
                ax,ay = a.center; bx1,by1,bx2,by2 = b.box
                aa = (a.box[2]-a.box[0])*(a.box[3]-a.box[1])
                ba = (b.box[2]-b.box[0])*(b.box[3]-b.box[1])
                if bx1<=ax<=bx2 and by1<=ay<=by2 and aa<ba and b.cls in ("plate","bowl","table"):
                    on.append((a.inst_id, b.inst_id)); break
        supported = {a for a,_ in on}
        for d in dets:
            if d.inst_id not in supported and obj_class(d.inst_id)!="table":
                on.append((d.inst_id, "table_1"))
        for d in dets: self.last_pixel[d.inst_id] = d.center
        return Scene(objects=ids, held=self.held, open_objects=[], on=on, inside=[])

                       
    def plan(self, scene, command):
        try:
            from larp_agent import LARPAgent
            result = LARPAgent().plan(command, scene)
            plan = [{"action":t.action,"object":t.object,"target":t.target}
                    for t in result["itg"].trios]
            print(f"    agent valid={result['valid']} rounds={result['rounds']}")
            if not result["valid"]:
                print(f"    WARNING: {result['errors']}")
            return plan
        except Exception as e:
            print(f"    [agent error: {e}]; stub plan")
            return [
                
                {"action":"pick","object":"bread_1","target":None},
                {"action":"put","object":"bread_1","target":"ground beef_1"},]
    
      

                               
    def verify(self, scene, trio):

        a, o = trio["action"], trio["object"]
        on = list(scene.on); inside = list(getattr(scene,"inside",[]))
        objs_on = lambda x: [s for s,t in on if t==x]
        cont_of = lambda x: next((t for s,t in inside if s==x), None)
        if a == "pick":
            if self.held is not None:
                return {"violation":True,"category":"occupancy","blocker":self.held}
            c = cont_of(o)
            if c and c not in scene.open_objects:
                return {"violation":True,"category":"containment","blocker":c}
            b = objs_on(o)
            if b: return {"violation":True,"category":"obstruction","blocker":b[0]}
        elif a == "put":
            if self.held != o:
                return {"violation":True,"category":"omitted_prerequisite","blocker":o}
        elif a == "slice":
            if self.held is None or obj_class(self.held) != "knife":
                return {"violation":True,"category":"omitted_prerequisite","blocker":"knife_1"}
        return {"violation":False,"category":"none","blocker":None}

    def repair_for(self, v, trio):
        c, b = v["category"], v["blocker"]
        if c == "obstruction":
            return [{"action":"pick","object":b,"target":None},
                    {"action":"put","object":b,"target":"clear_location"}]
        if c == "containment": return [{"action":"open","object":b,"target":None}]
        if c == "occupancy":
            return [{"action":"put","object":b,"target":"clear_location"}]
        if c == "omitted_prerequisite":
            if trio["action"]=="slice": return [{"action":"pick","object":"knife_1","target":None}]
            if trio["action"]=="put": return [{"action":"pick","object":trio["object"],"target":None}]
        return []

                                       
    def resolve_pixel(self, plan_id, dets):
        cls = obj_class(plan_id)
        cands = [d for d in dets if d.cls == cls]
        if not cands: return None
        if len(cands)==1: return cands[0].center
        last = self.last_pixel.get(plan_id)
        if last is None: return max(cands, key=lambda d: d.conf).center
        return min(cands, key=lambda d: (d.center[0]-last[0])**2+(d.center[1]-last[1])**2).center

                                                                      
                                                                        
                                                                           
                                                                           
                                                                              
                                                                                
                                                                            
                                                                           
                                                                            
    Z_MIN = -61.75                             
    Z_MAX = -37.0                                                         
    XY_LIMIT = 40.0                                                                      

    def pixel_to_delta_frame(self, u, v, plan_id, verbose=True):
        
        z_obj = self.cfg.z_obj.get(obj_class(plan_id), self.cfg.default_z_obj)
        cap = np.array(self.Delta.read_forward())                            
        robot = Camera.pixel_to_robot_coordinates(
            (u, v),
            z_obj=self.cfg.z_fom + z_obj,
            gripper=self.cfg.gripper_type,                        
            robot_capturing_coord=cap)
        x, y, z = float(robot[0]), float(robot[1]), float(robot[2])
        z -= (z_obj / 2.0)                                                         
        if verbose:
            print(f"      [calib] pixel ({u:.0f},{v:.0f})  --measured transform-->  "
                  f"Delta frame ({x:.2f}, {y:.2f}, {z:.2f})  [z_obj={z_obj}cm]")
                              
        msgs = []
        if not (self.Z_MIN <= z <= self.Z_MAX):
            msgs.append(f"z={z:.2f} outside [{self.Z_MIN}, {self.Z_MAX}]")
        if abs(x) > self.XY_LIMIT or abs(y) > self.XY_LIMIT:
            msgs.append(f"(x,y)=({x:.1f},{y:.1f}) exceeds +/-{self.XY_LIMIT}cm")
        if msgs:
            raise ValueError("grasp point outside Delta workspace: " + "; ".join(msgs))
        return x, y, z

    def to_robot(self, u, v, plan_id):
                                                                             
                                                
        return self.pixel_to_delta_frame(u, v, plan_id, verbose=True)

                                            
    def _safe_move(self, x, y, z, t):
        
        z_clamped = max(self.Z_MIN, min(self.Z_MAX, z))
        if abs(z_clamped - z) > 1e-6:
            print(f"      [Z-CLAMP] requested z={z:.2f} -> clamped to "
                  f"{z_clamped:.2f} (limits {self.Z_MIN}..{self.Z_MAX})")
        self.Delta.move_with_time(x, y, z_clamped, t)

    def adept_move(self, tx, ty, tz, from_capturing=False):
        
        off = self.cfg.above_offset
        travel_z = self.Z_MAX                                                                   
        approach_z = max(self.Z_MIN, min(self.Z_MAX, tz + off))                                
                                                                             
        self._safe_move(tx, ty, travel_z, 5)
                                                                                 
        self._safe_move(tx, ty, approach_z, 5)

    def go_capturing(self):
        cx, cy, cz = self.cfg.capturing_coord
        self._safe_move(cx, cy, cz, 5)

                                                          
    def confirm(self, frame, u, v, label):
        
                                                                              
        if self.cfg.auto_confirm:
            if self.cfg.live:
                raise RuntimeError("auto_confirm is forbidden with --live (safety).")
            print(f"      [auto-confirm (DRY TEST ONLY)] {label}")
            return True
                                                        
        vis = frame.copy() if frame is not None else None
        if vis is not None and u is not None:
            cv2.circle(vis, (int(u), int(v)), 20, (0, 0, 255), 3)
            cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0), cv2.MARKER_CROSS, 36, 2)
            cv2.putText(vis, f"{label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                                                                           
        gui_ok = True
        if vis is not None:
            try:
                cv2.imshow('Debug Zone', vis)
                cv2.waitKey(1)
            except Exception:
                gui_ok = False
                path = f"debug_zone_{int(time.time())}.png"
                cv2.imwrite(path, vis)
                print(f"      [Debug Zone] GUI unavailable; saved -> {path} (open it to inspect)")

                                                      
        print(f"\n      >>> CONFIRM MOTION: {label}")
        print(f"          grasp pixel = ({u}, {v})" if u is not None else "          (placement)")
        if gui_ok and vis is not None:
            print("          focus the 'Debug Zone' window, press [y]=go [n]=skip [Esc]=abort")
            while True:
                k = cv2.waitKeyEx(0)
                if k == ord('y'): return True
                if k == ord('n'): print("          skipped."); return False
                if k == 27: raise KeyboardInterrupt("operator aborted")
        else:
                                                
            ans = input("          type 'y' to MOVE, 'n' to skip, 'a' to abort: ").strip().lower()
            if ans == 'y': return True
            if ans == 'a': raise KeyboardInterrupt("operator aborted")
            print("          skipped."); return False

                                                    
    def grasp_point(self, plan_id, action, frame, dets):
        
        cands = self._sensetask_candidates(plan_id, frame, dets)
        center = self.resolve_pixel(plan_id, dets)
        if cands:
            print(f"      [SenseTask] {len(cands)} grasp candidate(s) for {plan_id}")
                                                                         
                                                   
            if center is not None:
                cands = list(cands) + [(center[0], center[1], 0.0)]
            return cands
                                                                     
        if center is not None:
            print(f"      [fallback] no SenseTask grasp for {plan_id}; "
                  f"using object center ({center[0]:.0f},{center[1]:.0f})")
            return [(center[0], center[1], 0.0)]
        print(f"      [skip] {plan_id}: no grasp and no detected center")
        return []

    def _load_sensetask(self):
        if self._sensetask is None:
            from sensetask.segmentation.semantic_grasp_generator import initialize_models
            print(f"      [SenseTask] loading from {self.cfg.sensetask_models_path} ...")
            self._sensetask = initialize_models(self.cfg.sensetask_models_path,
                                                model_type="mask2former")
        return self._sensetask

    def _find_det(self, plan_id, dets):
        
        cls = obj_class(plan_id)
        cands = [d for d in dets if d.cls == cls]
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0]
        last = self.last_pixel.get(plan_id)
        if last is None:
            return max(cands, key=lambda d: d.conf)
        return min(cands, key=lambda d: (d.center[0]-last[0])**2 + (d.center[1]-last[1])**2)

    def _roi_for(self, det, frame):
        
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = det.box
        m = self.cfg.roi_margin_cm * self.cfg.px_per_cm                       
        x0 = int(max(0, x1 - m)); y0 = int(max(0, y1 - m))
        x3 = int(min(W, x2 + m)); y3 = int(min(H, y2 + m))
        return x0, y0, x3, y3

    def _sensetask_candidates(self, plan_id, frame, dets):
        
        import cv2
        models = self._load_sensetask()
        from sensetask.segmentation.semantic_grasp_generator import generate_masked_grasps
        os.makedirs("st_out", exist_ok=True)

        det = self._find_det(plan_id, dets)
        if det is not None:
            x0, y0, x1, y1 = self._roi_for(det, frame)
            if (x1 - x0) < 10 or (y1 - y0) < 10:
                x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]
        else:
            print(f"      [ROI] {plan_id} not found -> full image")
            x0, y0, x1, y1 = 0, 0, frame.shape[1], frame.shape[0]

        crop = frame[y0:y1, x0:x1]
        crop_h, crop_w = crop.shape[:2]
        print(f"      [ROI] {plan_id}: crop ({x0},{y0})-({x1},{y1}) "
              f"size {crop_w}x{crop_h} (margin {self.cfg.roi_margin_cm}cm)")

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                                                
        pool = []
        try:
            grasps = generate_masked_grasps(rgb, models, model_type="mask2former",
                                            output_dir="st_out", raw_image=crop)
            if isinstance(grasps, dict):
                for task, poses in grasps.items():
                    for p in (poses or []):
                        pool.append(tuple(p))
            elif isinstance(grasps, (list, tuple)):
                pool = [tuple(p) for p in grasps]
        except Exception as e:
            print(f"      [SenseTask][task grasps error] {e}")

        crop_scale = 1.0                                                

                                                                           
        if not pool:
            print(f"      [SenseTask] no task grasp; using raw mask grasps")
            try:
                gg = models["grasp_generator"]
                raw_poses, _ = gg.return_grasp_poses(rgb)                        
                                                                                   
                msize = 256.0
                sx = crop_w / msize
                sy = crop_h / msize
                for p in (raw_poses or []):
                    px, py = float(p[0]) * sx, float(p[1]) * sy
                    th = float(p[3]) if len(p) > 3 else 0.0
                    pool.append((px, py, 0.0, th))
            except Exception as e:
                print(f"      [SenseTask][raw grasps error] {e}")

        if not pool:
            print(f"      [SenseTask] no grasps at all (task or raw) for {plan_id}")
            return []

                                                              
        full = []
        for p in pool:
            cx, cy = float(p[0]) + x0, float(p[1]) + y0
            theta = float(p[3]) if len(p) > 3 else 0.0
            full.append((cx, cy, theta))

        center = det.center if det is not None else self.resolve_pixel(plan_id, dets)
        def rank_key(g):
            if center is None: return 0.0
            return (g[0]-center[0])**2 + (g[1]-center[1])**2
        full.sort(key=rank_key)
        return full

                             
    def do_grasp(self, plan_id, action, frame, dets, from_capturing=False):
        candidates = self.grasp_point(plan_id, action, frame, dets)
        if not candidates:
            print(f"      [skip] no grasp for {plan_id}")
            return False

                                                                        
                                                                             
        x = y = z = theta = None
        chosen = None
        for idx, (u, v, th) in enumerate(candidates):
            try:
                gx, gy, gz = self.to_robot(u, v, plan_id)                        
            except ValueError as e:
                print(f"      [candidate {idx}] rejected: {e}")
                continue
            x, y, z, theta, chosen = gx, gy, gz, th, (u, v)
            print(f"      [candidate {idx}] OK -> using pixel ({u:.0f},{v:.0f}) "
                  f"theta={th:.0f}")
            break
        if chosen is None:
            print(f"      [skip] all {len(candidates)} grasp candidates for "
                  f"{plan_id} were outside the workspace. Not moving.")
            return False

        print(f"      robot: ({x:.2f}, {y:.2f}, {z:.2f})")
        self._record("grasp_start", obj=plan_id, x=round(x,2), y=round(y,2), z=round(z,2))
        self.adept_move(x, y, z, from_capturing=from_capturing)
        if not self.confirm(frame, chosen[0], chosen[1], f"grasp {plan_id}"): return False
        if abs(theta) > 1: self.Delta.rotate_gripper(theta)
        self.Delta.open_gripper()
        self._safe_move(x, y, z, 5)
        self.Delta.close_gripper()
        self._safe_move(x, y, z + self.cfg.above_offset, 5)
        self._record("grasp_done", obj=plan_id)
        return True

    def do_place(self, plan_id, target_id, frame, dets):
            held_h = self.cfg.z_obj.get(obj_class(plan_id), self.cfg.default_z_obj)
            if target_id in (None, "clear_location"):
                x, y, z = self.cfg.clear_location_xyz
            else:
                c = self.resolve_pixel(target_id, dets)
                if c is None:
                    print(f"      [warn] {target_id} not found -> clear_location")
                    x, y, z = self.cfg.clear_location_xyz
                else:
                    x, y, z = self.to_robot(c[0], c[1], target_id)
                    target_h = self.cfg.z_obj.get(obj_class(target_id), self.cfg.default_z_obj)
                                                                             
                    stack = self._stack_height.get(target_id, 0.0)
                    z += target_h + stack + self.cfg.release_clearance
                    print(f"      [stack] {target_id} has {stack:.1f}cm on it already")
            print(f"      place {plan_id} -> {target_id} at ({x:.2f},{y:.2f},{z:.2f}) (held_h={held_h}cm)")
            self._record("place_start", obj=plan_id, target=target_id)
            self.adept_move(x, y, z, from_capturing=False)
            if not self.confirm(frame, None, None, f"place {plan_id}->{target_id}"): return False
            self._safe_move(x, y, z, 5)
            time.sleep(0.3)
            self.Delta.open_gripper()
            time.sleep(0.3)
            self._safe_move(x, y, z + self.cfg.above_offset, 5)
                                                                               
            if target_id not in (None, "clear_location"):
                self._stack_height[target_id] = self._stack_height.get(target_id, 0.0) + held_h
            self._record("place_done", obj=plan_id, target=target_id)
            if not self.cfg.live and target_id in (None, "clear_location"):
                self._stub_removed = getattr(self, "_stub_removed", set()) | {plan_id}
            return True

                        
    def run(self, command):
        print(f"\n{'='*68}\n  LARP PRODUCTION | live={self.cfg.live} command=\"{command}\"\n{'='*68}")
        print("\n[init] capturing position...")
        self.go_capturing()

        print("\n[1] CAPTURE")
        frame = self.capture()

        print("\n[2] DETECT")
        dets = self.detect(frame)
        scene = self.build_scene(dets)
        print(f"    objects: {[d.inst_id for d in dets]}")
        print(f"    on: {scene.on}")

        print("\n[3] PLAN")
        plan = self.plan(scene, command)
        print(f"    ITG: {[(t['action'],t['object'],t.get('target')) for t in plan]}")

        i, iters, MAX = 0, 0, 100
        repair_counts = {}
        grasp_failures = {}                                                     
        MAX_GRASP_FAILS = 3                                                       
        from_cap = True
        try:
            while i < len(plan) and iters < MAX:
                iters += 1
                trio = plan[i]
                a, o, t = trio["action"], trio["object"], trio.get("target")
                print(f"\n[node {i}] ({a}, {o}, {t})  held={self.held}")

                v = self.verify(scene, trio)
                if v["violation"]:
                    fix = self.repair_for(v, trio)
                    key = f"{i}:{v['category']}:{v['blocker']}"
                    repair_counts[key] = repair_counts.get(key, 0) + 1
                    if fix and repair_counts[key] <= 2:
                        print(f"      ! {v['category']} ({v['blocker']}) -> {[(f['action'],f['object']) for f in fix]}")
                        plan[i:i] = fix; continue
                    elif fix:
                        print(f"      [warn] repeated repair; skipping")

                if a == "put":
                    if self.do_place(o, t, frame, dets): self.held = None
                elif a in GRASP_ACTIONS:
                    if a != "pick": print(f"      [note] '{a}' sub-action is placeholder")
                    if self.do_grasp(o, a, frame, dets, from_capturing=from_cap):
                        self.held = o
                        grasp_failures[o] = 0
                    else:
                                                                            
                                                                              
                                                                             
                                                                        
                        grasp_failures[o] = grasp_failures.get(o, 0) + 1
                        if grasp_failures[o] >= MAX_GRASP_FAILS:
                            print(f"      [give up] {o} could not be grasped after "
                                  f"{MAX_GRASP_FAILS} tries; skipping it and moving on.")
                                                                              
                            plan = [n for n in plan
                                    if not (n["object"] == o or n.get("target") == o)]
                            continue
                        print(f"      [grasp failed {grasp_failures[o]}/{MAX_GRASP_FAILS} "
                              f"for {o}] re-detecting and retrying...")
                        self.go_capturing()
                        frame = self.capture()
                        dets = self.detect(frame)
                        scene = self.build_scene(dets)
                        from_cap = True
                        continue
                    from_cap = False
                    if t and t != "clear_location":
                        if self.do_place(o, t, frame, dets): self.held = None
                else:
                    print(f"      [skip] unsupported '{a}'")

                                 
                self.go_capturing()
                frame = self.capture()
                dets = self.detect(frame)
                scene = self.build_scene(dets)
                from_cap = True
                print(f"      [re-detect] {[d.inst_id for d in dets]}")
                i += 1

            print(f"\n{'='*68}\n  PLAN COMPLETE ({i} nodes)\n{'='*68}")
        except KeyboardInterrupt as e:
            print(f"\n[ABORTED] {e}"); self.Delta.go_home()
        except RuntimeError as e:
            print(f"\n[STOPPED] {e}"); self.Delta.go_home()
        finally:
            self._save_log()
            if self.cap: self.cap.release()
            try: cv2.destroyAllWindows()
            except Exception: pass

    def _record(self, kind, **d): self.log.append({"t":round(time.time(),3),"kind":kind,**d})
    def _save_log(self):
        try: json.dump(self.log, open(self.cfg.log_path,"w"), indent=2); print(f"[log] {len(self.log)} events -> {self.cfg.log_path}")
        except Exception as e: print(f"[log] {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="make a sandwich")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--image", default=None)
    ap.add_argument("--auto-confirm", action="store_true",
                    help="DRY-TEST ONLY: skip confirmation prompts (refused with --live)")
    args = ap.parse_args()
    cfg = Config(live=args.live, test_image=args.image, auto_confirm=args.auto_confirm)
    LARPProduction(cfg).run(args.command)