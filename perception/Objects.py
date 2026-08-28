from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-world.pt")                          
model.set_classes(["knife", "cup", "plate", "bread", "Sliced tomato","patty","lettuce","fork"])

results = model.predict("F:/sensetask/SenseTask/YOLO/table.jpg", conf=0.25, save=True,
                         project="my_output", name="test1")
                              
results[0].show()                                              

boxes = results[0].boxes
names = results[0].names

detected = []                                                                       
counts = {}
for b in boxes:
    cls = names[int(b.cls)]
    counts[cls] = counts.get(cls, 0) + 1
    x1, y1, x2, y2 = b.xyxy[0].tolist()
    detected.append({
        "id": f"{cls}_{counts[cls]}",                                
        "class": cls,
        "center": ((x1 + x2) / 2, (y1 + y2) / 2),
        "box": (x1, y1, x2, y2),
    })

object_ids = [d["id"] for d in detected]                                           