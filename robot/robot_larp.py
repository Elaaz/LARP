

import cv2
import numpy as np

import delta_manager.camera as Camera
from delta_manager.delta_manager import DeltaManager


                                                                              
                                                      
                                                                              
ENABLE_MOTION = False                                                           
REQUIRE_CONFIRM = True                                             

                                                                             
                                                                                
                                                                    
APPROACH = 10.0                                                    
MOVE_T = 3                                                
DESCEND_T = 3                                                 

                                                                               
Z_OBJ = {"knife": 1.9, "cup": 8.0, "plate": 2.0, "bread": 4.0,
         "tomato": 5.0, "apple": 7.0, "bowl": 6.0}
DEFAULT_Z_OBJ = 2.0

                                                                     
                                                                            
CLEAR_LOCATION_XYZ = (15.0, 15.0, -30.0)                               

SENSETASK_OBJECTS = {"knife", "cup"}
ACTION_TO_TASK = {"slice": "Tool Usage", "pour": "Pouring",
                  "open": "Opening_Closing", "close": "Opening_Closing",
                  "pick": "Stabilizing"}

import re
def obj_class(oid): return re.sub(r"_\d+$", "", oid)


                                                                              
              
                                                                              
Delta = DeltaManager()


                                                                              
                                                    
                                                                              
def send_gripper(cmd: str):
    
    print(f"    [gripper] {cmd}")
    if not ENABLE_MOTION:
        return
                                                                            
    sender = getattr(Delta, "send_gripper_command", None)
    if callable(sender):
        sender(cmd)
    else:
        raise NotImplementedError(
            "Wire send_gripper() to your gripper transport before enabling motion.")
                                                                             

def gripper_open():  send_gripper("cg1")
def gripper_close(): send_gripper("cg2")
def gripper_rotate(rx=0, ry=0, rz=0):
    send_gripper(f"cr{int(round(rx))}-{int(round(ry))}-{int(round(rz))}")


                                                                              
                
                                                                              
def _move(x, y, z, t=MOVE_T):
    print(f"    [move] -> ({x:.2f}, {y:.2f}, {z:.2f})")
    if ENABLE_MOTION:
        Delta.move_with_time(x, y, z, t)
        Delta.wait_till_done_robot()

def move_above(x, y, z):
    
    _move(x, y, z + APPROACH, MOVE_T)

def descend_to(x, y, z):
    _move(x, y, z, DESCEND_T)

def lift_from(x, y, z):
    _move(x, y, z + APPROACH, MOVE_T)


                                                                              
                
                                                                              
def to_robot(u, v, object_id):
    z_obj = Z_OBJ.get(obj_class(object_id), DEFAULT_Z_OBJ)
    x, y, z = Camera.pixel_to_robot_coordinates(
        (u, v), z_obj=z_obj, gripper='2f85',
        robot_capturing_coord=np.array(Delta.read_forward()))
    z -= (z_obj / 2.0)
    return float(x), float(y), float(z)


                                                                              
                                                                      
                                                                              
                                                                          
                                                                       

def detect_object_pixel(object_id, frame, detections):
    
    for d in detections:
        if getattr(d, "inst_id", None) == object_id:
            return d.center
    return None

def sensetask_grasp(object_id, task, frame):
    
    return None                                                      


def resolve_grasp(object_id, action, frame, detections):
    
    cls = obj_class(object_id)
    if cls in SENSETASK_OBJECTS:
        task = ACTION_TO_TASK.get(action, "Stabilizing")
        g = sensetask_grasp(object_id, task, frame)
        if g is not None:
            u, v, theta = g
            print(f"    [grasp] {object_id}: SenseTask task='{task}' "
                  f"-> pixel ({u:.0f},{v:.0f}) theta={theta:.0f}")
            return u, v, theta
        print(f"    [grasp] {object_id}: SenseTask returned nothing -> generic center")
    c = detect_object_pixel(object_id, frame, detections)
    if c is None:
        return None
    u, v = c
    print(f"    [grasp] {object_id}: generic center ({u:.0f},{v:.0f})")
    return u, v, 0.0


def resolve_target_xyz(target_id, frame, detections):
    
    if target_id in (None, "clear_location"):
        print(f"    [target] clear_location -> {CLEAR_LOCATION_XYZ}")
        return CLEAR_LOCATION_XYZ
    c = detect_object_pixel(target_id, frame, detections)
    if c is None:
        print(f"    [target] {target_id} not detected -> using clear_location")
        return CLEAR_LOCATION_XYZ
    u, v = c
    return to_robot(u, v, target_id)


                                                                              
                     
                                                                              
def confirm_location(frame, u, v, label):
    
    if not REQUIRE_CONFIRM:
        return True
    vis = frame.copy()
    if u is not None:
        cv2.circle(vis, (int(u), int(v)), 18, (0, 0, 255), 3)
        cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 0),
                       cv2.MARKER_CROSS, 30, 2)
    cv2.putText(vis, f"{label}  [y]=grasp  [n]=skip  [Esc]=abort",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.imshow('confirm', vis)
    print(f"    [confirm] {label}: now 10cm above. y=approve, n=skip, Esc=abort")
    while True:
        k = cv2.waitKeyEx(0)
        if k == ord('y'): return True
        if k == ord('n'): return False
        if k == 27: raise KeyboardInterrupt("operator aborted")


                                                                              
                    
                                                                              
def do_grasp(object_id, action, frame, detections):
    
    g = resolve_grasp(object_id, action, frame, detections)
    if g is None:
        print(f"    [skip] could not resolve grasp for {object_id}")
        return False
    u, v, theta = g
    x, y, z = to_robot(u, v, object_id)
    move_above(x, y, z)                                      
    if not confirm_location(frame, u, v, f"grasp {object_id}"):
        print(f"    [skip] {object_id} not approved")
        return False
    gripper_rotate(0, 0, theta)                                         
    gripper_open()                                   
    descend_to(x, y, z)                                    
    gripper_close()                                  
    lift_from(x, y, z)                                
    return True

def do_place(object_id, target_id, frame, detections):
    
    x, y, z = resolve_target_xyz(target_id, frame, detections)
    move_above(x, y, z)                                          
    if not confirm_location(frame, None, None, f"place {object_id} -> {target_id}"):
        print(f"    [skip] placement not approved (still holding {object_id})")
        return False
    descend_to(x, y, z)
    gripper_open()                                          
    lift_from(x, y, z)
    return True


                                                                              
                
                                                                              
def execute_plan(plan, frame, detections):
    
    held = None
    for i, trio in enumerate(plan):
        a, o, t = trio["action"], trio["object"], trio.get("target")
        print(f"\n[{i}] ({a}, {o}, {t})  held={held}")
        if a == "put":
            if held != o:
                print(f"    [warn] plan asks to put {o} but holding {held}; "
                      f"grasping {o} first")
                if not do_grasp(o, "pick", frame, detections):
                    continue
                held = o
            if do_place(o, t, frame, detections):
                held = None
        elif a in ("pick", "slice", "open", "close", "pour"):
                                                                          
                                                                              
                                                                          
            if a != "pick":
                print(f"    [note] '{a}' physical sub-action is a placeholder; "
                      f"executing the grasp only.")
            if do_grasp(o, a, frame, detections):
                held = o
                                                         
            if t and t not in (None, "clear_location"):
                if do_place(o, t, frame, detections):
                    held = None
        else:
            print(f"    [skip] unsupported action '{a}'")
    print("\n[plan complete]")


                                                                              
                                                          
                                                                              
def run(plan=None):
    
    cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4000)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 4000)
    print(f"ENABLE_MOTION={ENABLE_MOTION}  REQUIRE_CONFIRM={REQUIRE_CONFIRM}  "
          f"APPROACH={APPROACH}")

    _, frame = cap.read()
    frame = Camera.undistort(frame)                                               

    detections = []
    if plan is None:
                                                                               
        try:
            from perception import build_scene
            scene, detections = build_scene(frame,
                ["knife", "cup", "plate", "bread", "tomato", "apple", "bowl"])
            from larp_agent import LARPAgent
            command = input("command> ") or "make a sandwich"
            result = LARPAgent().plan(command, scene)
            plan = [{"action": t.action, "object": t.object, "target": t.target}
                    for t in result["itg"].trios]
                                                                                      
        except Exception as e:
            print(f"[pipeline unavailable: {e}] — pass a plan= explicitly.")
            cap.release(); return

    try:
        execute_plan(plan, frame, detections)
    except KeyboardInterrupt as e:
        print(f"\n[ABORTED] {e}")
        if ENABLE_MOTION:
            Delta.go_home()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
                                                                                    
    demo_plan = [
        {"action": "pick", "object": "knife_1", "target": None},
        {"action": "put",  "object": "knife_1", "target": "plate_1"},
        {"action": "pick", "object": "apple_1", "target": None},
        {"action": "put",  "object": "apple_1", "target": "clear_location"},
    ]
    run(plan=demo_plan)
