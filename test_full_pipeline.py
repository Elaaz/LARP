

import os, sys, argparse, re

                                                                              
_ROOT = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "perception", "strategic", "grasping", "robot", "model", "domain"]:
    p = os.path.join(_ROOT, sub)
    if os.path.isdir(p) or sub == "":
        sys.path.insert(0, p)

                                                                               
SENSETASK_REPO = os.path.join(_ROOT, "SenseTask")
MODELS_PATH    = os.path.join(_ROOT, "SenseTask", "models")
sys.path.insert(0, SENSETASK_REPO)

ACTION_TO_TASK = {"slice": "Tool Usage", "pour": "Pouring",
                  "open": "Opening_Closing", "close": "Opening_Closing",
                  "pick": "Stabilizing"}
SENSETASK_OBJECTS = {"knife", "cup"}
GRASP_ACTIONS = {"pick", "slice", "open", "close", "pour"}

def obj_class(oid): return re.sub(r"_\d+$", "", oid)


                                                                              
               
                                                                              
def run_perception(image, classes):
    from perception import build_scene
    scene, dets = build_scene(image, classes)
    print("\n[1] PERCEPTION")
    print("    objects:", scene.objects)
    print("    on     :", scene.on)
    return scene, dets


                                                                              
                     
                                                                              
def run_agent(command, scene):
    from larp_agent import LARPAgent
    print("\n[2] AGENT (planning)")
    result = LARPAgent().plan(command, scene)
    plan = [{"action": t.action, "object": t.object, "target": t.target}
            for t in result["itg"].trios]
    print(f"    valid={result['valid']} rounds={result['rounds']}")
    for i, t in enumerate(plan):
        print(f"    {i}: ({t['action']}, {t['object']}, {t.get('target')})")
    if not result["valid"]:
        print("    WARNING residual errors:", result["errors"])
    return plan


                                                                              
                                                                        
                                                                              
def _affordances(cls):
    t = {"plate": ["surface","container"], "bowl": ["container"], "cup": ["container"],
         "knife": ["tool"], "fork": ["tool"], "fridge": ["openable","container"]}
    return t.get(cls, [])

def verify(scene, trio, held):
    a, o = trio["action"], trio["object"]
    on = list(scene.on); inside = list(getattr(scene, "inside", []))
    objs_on = lambda x: [s for s, tt in on if tt == x]
    cont_of = lambda x: next((tt for s, tt in inside if s == x), None)
    if a == "pick":
        if held is not None:
            return {"violation": True, "category": "occupancy", "blocker": held}
        c = cont_of(o)
        if c and c not in scene.open_objects:
            return {"violation": True, "category": "containment", "blocker": c}
        b = objs_on(o)
        if b: return {"violation": True, "category": "obstruction", "blocker": b[0]}
    elif a == "put":
        if held != o:
            return {"violation": True, "category": "omitted_prerequisite", "blocker": o}
    elif a == "slice":
        if held is None or "tool" not in _affordances(obj_class(held)):
            return {"violation": True, "category": "omitted_prerequisite", "blocker": "knife_1"}
    return {"violation": False, "category": "none", "blocker": None}

def repair_for(v, trio):
    c, b = v["category"], v["blocker"]
    if c == "obstruction": return [{"action":"pick","object":b,"target":None},
                                   {"action":"put","object":b,"target":"clear_location"}]
    if c == "containment": return [{"action":"open","object":b,"target":None}]
    if c == "occupancy":   return [{"action":"put","object":b,"target":"clear_location"}]
    if c == "omitted_prerequisite":
        if trio["action"] == "slice": return [{"action":"pick","object":"knife_1","target":None}]
        if trio["action"] == "put":   return [{"action":"pick","object":trio["object"],"target":None}]
    return []

def apply_effect(scene, trio, held):
    a, o, t = trio["action"], trio["object"], trio.get("target")
    scene.on = [(s, tt) for s, tt in scene.on if s != o]
    if a == "pick": held = o
    elif a == "put":
        held = None
        if t and t != "clear_location": scene.on.append((o, t))
    return held

def verify_and_repair(scene, plan):
    print("\n[3] LARP VERIFY + REPAIR")
    import copy
    sc = copy.deepcopy(scene); held = getattr(sc, "held", None)
    plan = list(plan); out = []; i = it = 0
    while i < len(plan) and it < 100:
        it += 1; trio = plan[i]
        v = verify(sc, trio, held)
        if v["violation"]:
            fix = repair_for(v, trio)
            if fix:
                print(f"    ! ({trio['action']},{trio['object']}): {v['category']} "
                      f"-> insert {[(f['action'],f['object']) for f in fix]}")
                plan[i:i] = fix; continue
        out.append(trio); held = apply_effect(sc, trio, held); i += 1
    print("    repaired plan:")
    for j, t in enumerate(out):
        print(f"    {j}: ({t['action']}, {t['object']}, {t.get('target')})")
    return out


                                                                              
                                                                      
                                                                              
_ST = {"models": None, "grasps_cache": {}}

def load_sensetask():
    from sensetask.segmentation.semantic_grasp_generator import initialize_models
    print(f"    loading SenseTask from {MODELS_PATH} ...")
    _ST["models"] = initialize_models(MODELS_PATH, model_type="mask2former")
    print("    SenseTask loaded.")

def sensetask_grasps_for_image(image_path, out_dir="st_out"):
    
    import cv2
    from sensetask.segmentation.semantic_grasp_generator import generate_masked_grasps
    os.makedirs(out_dir, exist_ok=True)
    raw = cv2.imread(image_path)
    rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    grasps = generate_masked_grasps(rgb, _ST["models"], model_type="mask2former",
                                    output_dir=out_dir, raw_image=raw)
    _ST["grasps_cache"] = grasps
    return grasps

def get_grasp(trio, image_path, use_real_sensetask):
    
    a, o = trio["action"], trio["object"]
    if a == "put" or a not in GRASP_ACTIONS:
        return {"route": "place/none", "grasp": None}
    if obj_class(o) in SENSETASK_OBJECTS:
        task = ACTION_TO_TASK.get(a, "Stabilizing")
        if use_real_sensetask:
            if not _ST["grasps_cache"]:
                sensetask_grasps_for_image(image_path)
            poses = _ST["grasps_cache"].get(task, [])
            grasp = poses[0] if poses else None
            return {"route": f"SenseTask[{task}]", "grasp": grasp,
                    "note": None if grasp else "no grasp returned (check image/task)"}
        return {"route": f"SenseTask[{task}]", "grasp": "SE2(stub)"}
    return {"route": "generic", "grasp": "GENERIC(stub)"}

def run_grasp_loop(plan, image_path, use_real_sensetask):
    print("\n[4] SENSETASK LOOP (per trio; no robot motion)")
    if use_real_sensetask:
        load_sensetask()
    for i, trio in enumerate(plan):
        g = get_grasp(trio, image_path, use_real_sensetask)
        line = f"    {i}: ({trio['action']}, {trio['object']}, {trio.get('target')})  -> {g['route']}"
        if g["grasp"] is not None:
            line += f"  grasp={g['grasp']}"
        if g.get("note"):
            line += f"  [{g['note']}]"
        print(line)


                                                                              
      
                                                                              
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--command", default="make a sandwich")
    ap.add_argument("--classes", nargs="+",
                    default=["knife","cup","plate","bread","tomato","apple","bowl","meat","fork"])
    ap.add_argument("--real_sensetask", action="store_true",
                    help="actually run SenseTask (needs the repo + models)")
    args = ap.parse_args()

    print(f"=== FULL PIPELINE TEST ===  image={args.image}  command=\"{args.command}\"")
    scene, dets = run_perception(args.image, args.classes)
    plan = run_agent(args.command, scene)
    repaired = verify_and_repair(scene, plan)
    run_grasp_loop(repaired, args.image, args.real_sensetask)
    print("\n=== DONE (no robot motion) ===")

if __name__ == "__main__":
    main()
