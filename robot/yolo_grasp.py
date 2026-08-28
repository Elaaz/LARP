

import cv2
import numpy as np

                                             
import delta_manager.camera as Camera
from delta_manager.delta_manager import DeltaManager

              
from ultralytics import YOLOE


                                                                              
        
                                                                              
ENABLE_MOTION = False                                                         
WEIGHTS = "yoloe-v8s-seg.pt"
NAMES = ["knife", "cup", "plate", "bread", "tomato", "apple", "bowl"]
CONF, IOU = 0.45, 0.5

                                                                              
                                                                           
                                                                      
Z_OBJ = {"knife": 1.9, "cup": 8.0, "plate": 2.0, "bread": 4.0,
         "tomato": 5.0, "apple": 7.0, "bowl": 6.0}
DEFAULT_Z_OBJ = 2.0

CAM_INDEX = 2
WIDTH = HEIGHT = 4000


                                                                              
       
                                                                              
Delta = DeltaManager()
try:
    Delta.connect_gripper()
except Exception as e:
    print(f"[warn] connect_gripper failed ({e}); gripper commands may not work")

model = YOLOE(WEIGHTS)
model.set_classes(NAMES, model.get_text_pe(NAMES))


def pixel_to_robot(u, v, cls):
    
    z_obj = Z_OBJ.get(cls, DEFAULT_Z_OBJ)
    x, y, z = Camera.pixel_to_robot_coordinates(
        (u, v),
        z_obj=z_obj,
        gripper='2f85',
        robot_capturing_coord=np.array(Delta.read_forward())
    )
    z -= (z_obj / 2.0)
    return x, y, z, z_obj


def detect_best(frame):
    
    results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return None
    b = max(boxes, key=lambda b: float(b.conf))
    cls = results[0].names[int(b.cls)]
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    return cls, (x1 + x2) / 2.0, (y1 + y2) / 2.0, float(b.conf)


def grasp(x, y, z):
    
    Delta.delta_open_gripper()
    Delta.move_with_time(x, y, z + 6, 3)                                       
    Delta.move_with_time(x, y, z + 1, 3)                           
    Delta.move_with_time(x, y, z, 2)                                   
    Delta.wait_till_done_robot()
    Delta.delta_close_gripper()                    
    Delta.move_with_time(x, y, z + 6, 3)          
    Delta.wait_till_done_robot()


def handle_detection(frame, allow_motion):
    det = detect_best(frame)
    if det is None:
        print("  nothing detected"); return
    cls, u, v, conf = det
    x, y, z, z_obj = pixel_to_robot(u, v, cls)
    print(f"  detected {cls} (conf {conf:.2f}) at pixel ({u:.0f},{v:.0f})")
    print(f"  -> robot ({x:.2f}, {y:.2f}, {z:.2f})  [z_obj={z_obj}cm]")
    if not allow_motion:
        print("  [coordinates-only] ENABLE_MOTION is False — not moving.")
        return
    print("  moving + grasping ...")
    grasp(x, y, z)
    print("  grasp sequence complete.")


                                                                              
           
                                                                              
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    print(f"camera {int(cap.get(3))}x{int(cap.get(4))} | ENABLE_MOTION={ENABLE_MOTION}")
    print("keys: 'd'=detect only (never moves)  'y'=detect+grasp (moves if enabled)")
    print("      'o'/'c'=gripper open/close  'h'=home  'f'=print pose  Esc=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[warn] camera read failed"); continue
        frame = Camera.undistort(frame)                                      
        cv2.imshow('yolo_grasp', frame)

        key = cv2.waitKeyEx(1)
        if key == 27:                               
            break
        elif key == ord('d'):                                            
            print("[detect]")
            handle_detection(frame, allow_motion=False)
        elif key == ord('y'):                                               
            print("[detect + grasp]")
            handle_detection(frame, allow_motion=ENABLE_MOTION)
        elif key == ord('o'):
            Delta.delta_open_gripper()
        elif key == ord('c'):
            Delta.delta_close_gripper()
        elif key == ord('h'):
            Delta.go_home()
        elif key == ord('f'):
            print("pose:", Delta.read_forward())

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
