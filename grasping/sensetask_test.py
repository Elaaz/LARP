

import os, sys, argparse
from copy import deepcopy

                                                                              
                                                
                                                                              
SENSETASK_REPO = r"./SenseTask"                                                   
MODELS_PATH    = r"./models"                                                                            

sys.path.insert(0, os.path.abspath(SENSETASK_REPO))

                                                                              
                                                                                   
                                                                              
ACTION_TO_TASK = {
    "slice": "Tool Usage",
    "pour":  "Pouring",
    "open":  "Opening_Closing",
    "close": "Opening_Closing",
    "pick":  "Stabilizing",
}
SENSETASK_OBJECTS = {"knife", "cup"}
GRASP_ACTIONS = {"pick", "slice", "open", "close", "pour"}

def object_category(obj_id): return obj_id.rsplit("_", 1)[0]


                                                                              
                                                                            
                                                           
                                                                              
def affordances(cls):
    t = {"plate": ["graspable","surface","container"], "bowl": ["graspable","container"],
         "cup": ["graspable","container"], "knife": ["graspable","tool"],
         "fridge": ["openable","container"], "table": ["surface"],
         "milk": ["graspable","container"]}
    return t.get(cls, ["graspable"])

class W:
    def __init__(self, g):
        self.cls, self.held, self.openset, self.on, self.inside = {}, set(), set(), set(), set()
        for n in g["nodes"]:
            self.cls[n["id"]] = n["type"]
            if n["state"].get("held"): self.held.add(n["id"])
            if n["state"].get("open"): self.openset.add(n["id"])
        for e in g["edges"]:
            if e["relation"] == "on": self.on.add((e["subject"], e["object"]))
            if e["relation"] == "in": self.inside.add((e["subject"], e["object"]))
    def objects_on(self, i): return sorted(a for a, b in self.on if b == i)
    def container_of(self, i): return next((b for a, b in self.inside if a == i), None)
    def held_obj(self): return next(iter(self.held), None)

def verify(g, trio):
    w = W(g); a, o = trio["action"], trio["object"]
    if a == "pick":
        h = w.held_obj()
        if h: return dict(violation=True, category="occupancy", blocker=h)
        c = w.container_of(o)
        if c and c not in w.openset: return dict(violation=True, category="containment", blocker=c)
        on = w.objects_on(o)
        if on: return dict(violation=True, category="obstruction", blocker=on[0])
    elif a == "put" and o not in w.held:
        return dict(violation=True, category="omitted_prerequisite", blocker=o)
    elif a == "slice":
        h = w.held_obj()
        if h is None or "tool" not in affordances(w.cls.get(h, "")):
            return dict(violation=True, category="omitted_prerequisite", blocker="knife")
    return dict(violation=False, category="none", blocker=None)

def repair_for(v, trio):
    c, b = v["category"], v["blocker"]
    if c == "obstruction":  return [dict(action="pick", object=b, target=None),
                                    dict(action="put", object=b, target="clear_location")]
    if c == "containment":  return [dict(action="open", object=b, target=None)]
    if c == "occupancy":    return [dict(action="put", object=b, target="clear_location")]
    if c == "omitted_prerequisite":
        if trio["action"] == "slice": return [dict(action="pick", object="knife_1", target=None)]
        if trio["action"] == "put":   return [dict(action="pick", object=trio["object"], target=None)]
    return []

def apply_effect(g, trio):
    g = deepcopy(g); a, o, t = trio["action"], trio["object"], trio.get("target")
    nodes = {n["id"]: n for n in g["nodes"]}
    if a == "pick":
        if o in nodes: nodes[o]["state"]["held"] = True
        g["edges"] = [e for e in g["edges"] if e["subject"] != o]
    elif a == "put":
        if o in nodes: nodes[o]["state"]["held"] = False
        if t and t != "clear_location":
            g["edges"].append({"subject": o, "relation": "on", "object": t})
    elif a == "open":
        if o in nodes: nodes[o]["state"]["open"] = True
    elif a == "slice":
        if o in nodes: nodes[o]["state"]["sliced"] = True
    return g

def verify_and_repair(g0, plan, max_iters=20):
    g, plan, out, i, it = deepcopy(g0), list(plan), [], 0, 0
    while i < len(plan) and it < max_iters:
        it += 1
        v = verify(g, plan[i])
        if v["violation"]:
            fix = repair_for(v, plan[i])
            if fix: plan[i:i] = fix; continue
        out.append(plan[i]); g = apply_effect(g, plan[i]); i += 1
    return out


                                                                              
                                                                   
                                                                              
_ST = {"models": None, "grasps": None}

def load_sensetask():
    from sensetask.segmentation.semantic_grasp_generator import initialize_models
    print(f"[SenseTask] loading mask2former + skeleton generator from {MODELS_PATH} ...")
    _ST["models"] = initialize_models(MODELS_PATH, model_type="mask2former")
    print("[SenseTask] loaded.")

def run_sensetask_on_image(image_path, output_dir="./st_output"):
    
    import cv2
    from sensetask.segmentation.semantic_grasp_generator import generate_masked_grasps
    os.makedirs(output_dir, exist_ok=True)
    raw = cv2.imread(image_path)
    assert raw is not None, f"could not read image: {image_path}"
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    _ST["grasps"] = generate_masked_grasps(rgb, _ST["models"],
                                           model_type="mask2former",
                                           output_dir=output_dir, raw_image=raw)
    print("[SenseTask] grasps per task:",
          {k: len(v) for k, v in _ST["grasps"].items()})

def real_sensetask(obj_id, task):
    
    poses = _ST["grasps"].get(task, [])
    if not poses:
        print(f"    [SENSETASK] {obj_id} task='{task}' -> NO GRASP FOUND (check image/task)")
        return None
    g = poses[0]                                                         
    print(f"    [SENSETASK] {obj_id} task='{task}' -> grasp (x={g[0]:.0f}, y={g[1]:.0f}, w={g[2]:.0f}, th={g[3]:.0f})")
    return g

def generic_grasp(obj_id):
    print(f"    [GENERIC ] {obj_id} -> stable grasp (stub)"); return "GENERIC_GRASP"

def execute(trio, grasp):
    tag = f" grasp={grasp}" if grasp is not None else ""
    print(f"    [EXECUTE ] ({trio['action']}, {trio['object']}, {trio.get('target')}){tag}")

def run_plan(plan):
    for trio in plan:
        a = trio["action"]
        if a == "put" or a not in GRASP_ACTIONS:
            execute(trio, None); continue
        if object_category(trio["object"]) in SENSETASK_OBJECTS:
            grasp = real_sensetask(trio["object"], ACTION_TO_TASK[a])
        else:
            grasp = generic_grasp(trio["object"])
        execute(trio, grasp)


                                                                              
                
                                                                              
def node(i, t, **st):
    s = {"held": False, "open": None, "sliced": False, "clear": True}; s.update(st)
    return {"id": i, "type": t, "state": s}

CASES = [
 dict(name="obstruction",
      graph={"nodes":[node("plate_1","plate",clear=False),node("apple_1","apple"),node("table_1","table",clear=False)],
             "edges":[{"subject":"apple_1","relation":"on","object":"plate_1"},
                      {"subject":"plate_1","relation":"on","object":"table_1"}]},
      plan=[dict(action="pick",object="plate_1",target=None),
            dict(action="put",object="plate_1",target="table_1")],
      expect=[("pick","apple_1"),("put","apple_1")]),
 dict(name="containment",
      graph={"nodes":[node("fridge_1","fridge",open=False,clear=False),node("milk_1","milk"),node("table_1","table",clear=False)],
             "edges":[{"subject":"milk_1","relation":"in","object":"fridge_1"}]},
      plan=[dict(action="pick",object="milk_1",target=None),
            dict(action="put",object="milk_1",target="table_1")],
      expect=[("open","fridge_1")]),
 dict(name="occupancy",
      graph={"nodes":[node("cup_1","cup",held=True),node("knife_1","knife"),node("table_1","table",clear=False)],"edges":[]},
      plan=[dict(action="pick",object="knife_1",target=None)],
      expect=[("put","cup_1")]),
 dict(name="omitted_prerequisite_slice",
      graph={"nodes":[node("tomato_1","tomato"),node("knife_1","knife"),node("table_1","table",clear=False)],"edges":[]},
      plan=[dict(action="slice",object="tomato_1",target=None)],
      expect=[("pick","knife_1")]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="RGB test image with knife/cup on a table")
    ap.add_argument("--output_dir", default="./st_output")
    args = ap.parse_args()

    load_sensetask()
    run_sensetask_on_image(args.image, args.output_dir)                            

    passed = 0
    for c in CASES:
        print("="*70); print("CASE:", c["name"]); print("="*70)
        print("  flawed plan:", [(t["action"], t["object"]) for t in c["plan"]])
        rep = verify_and_repair(c["graph"], c["plan"])
        print("  repaired   :", [(t["action"], t["object"]) for t in rep])
        ok = all(e in [(t["action"], t["object"]) for t in rep] for e in c["expect"])
        print(f"  repair check: {'PASS' if ok else 'FAIL'}")
        print("  --- execution trace ---")
        run_plan(rep)
        passed += ok; print()
    print("="*70)
    print(f"RESULT: {passed}/{len(CASES)} repair checks passed")
    print(f"SenseTask visualizations saved in {args.output_dir} — open them and confirm")
    print("the 'Tool Usage' / 'Stabilizing' grasps sit on the knife/cup regions.")

if __name__ == "__main__":
    main()
