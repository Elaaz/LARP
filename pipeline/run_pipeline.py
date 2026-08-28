

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

                                                                           
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

                                                                              
                                                              
                                                                              
USE_REAL_PERCEPTION = False                                                 
USE_REAL_PLANNER    = False                                                 
USE_REAL_VERIFIER   = False                                                     
USE_REAL_SENSETASK  = False                                               
USE_REAL_ROBOT      = False                                                    

DEFAULT_CLASSES = ["knife", "cup", "plate", "bread", "tomato", "apple",
                   "bowl", "meat", "lettuce"]
SENSETASK_OBJECTS = {"knife", "cup"}
ACTION_TO_TASK = {"slice": "Tool Usage", "pour": "Pouring",
                  "open": "Opening_Closing", "close": "Opening_Closing",
                  "pick": "Stabilizing"}
GRASP_ACTIONS = {"pick", "slice", "open", "close", "pour"}


def banner(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)

def obj_class(oid: str) -> str:
    import re
    return re.sub(r"_\d+$", "", oid)


                                                                              
                                                                             
                                                                         
                                                                              
@dataclass
class SceneLite:
    objects: List[str]
    held: Optional[str] = None
    open_objects: List[str] = field(default_factory=list)
    on: List = field(default_factory=list)
    inside: List = field(default_factory=list)
    def exists(self, o): return o in self.objects
    def objects_on(self, o): return [a for a, b in self.on if b == o]
    def container_of(self, o):
        return next((b for a, b in self.inside if a == o), None)


                                                                              
                      
                                                                              
def stage_perception(image: str, classes: List[str]):
    banner("STAGE 1 — PERCEPTION (image -> scene graph)")
    if USE_REAL_PERCEPTION:
        from perception import build_scene
        scene, dets = build_scene(image, classes)
        print(f"  detected {len(dets)} objects from {image}")
    else:
                                                                           
                                                                           
        scene = SceneLite(
            objects=["plate_1", "apple_1", "knife_1", "tomato_1",
                     "bread_1", "table_1"],
            on=[("apple_1", "plate_1"), ("plate_1", "table_1"),
                ("knife_1", "table_1"), ("tomato_1", "table_1"),
                ("bread_1", "table_1")])
        print("  [stub] using a canned scene (apple on plate, knife/tomato/bread on table)")
    print(f"  objects: {scene.objects}")
    print(f"  on     : {scene.on}")
    print(f"  inside : {getattr(scene, 'inside', [])}")
    return scene


                                                                              
                    
                                                                              
def stage_planning(command: str, scene) -> List[Dict]:
    banner("STAGE 2 — PLANNING (command -> ITG of trios)")
    print(f"  command: \"{command}\"")
    if USE_REAL_PLANNER:
        from larp_agent import LARPAgent
        agent = LARPAgent()
        result = agent.plan(command, scene)
        plan = [t.as_dict() if hasattr(t, "as_dict") else
                {"action": t.action, "object": t.object, "target": t.target}
                for t in result["itg"].trios]
        print(f"  agent valid={result['valid']} (repair rounds={result['rounds']})")
        if not result["valid"]:
            print(f"  WARNING: residual errors: {result['errors']}")
    else:
                                                                              
                                                                         
                                       
        plan = [
            {"action": "slice", "object": "tomato_1", "target": None},                   
            {"action": "pick",  "object": "plate_1",  "target": None},                      
            {"action": "put",   "object": "plate_1",  "target": "table_1"},
        ]
        print("  [stub] using a deliberately flawed plan (missing knife pick; plate obstructed)")
    for i, t in enumerate(plan):
        print(f"    {i}: ({t['action']}, {t['object']}, {t.get('target')})")
    return plan


                                                                              
                           
                                                                              
def _rule_verify(scene, trio, held):
    
    a, o = trio["action"], trio["object"]
    on = list(scene.on); inside = list(getattr(scene, "inside", []))
    def objects_on(x): return [s for s, t in on if t == x]
    def container_of(x): return next((t for s, t in inside if s == x), None)
    if a == "pick":
        if held is not None:
            return {"violation": True, "category": "occupancy", "blocker": held}
        c = container_of(o)
        if c and c not in scene.open_objects:
            return {"violation": True, "category": "containment", "blocker": c}
        blk = objects_on(o)
        if blk:
            return {"violation": True, "category": "obstruction", "blocker": blk[0]}
    elif a == "put":
        if held != o:
            return {"violation": True, "category": "omitted_prerequisite", "blocker": o}
    elif a == "slice":
        if held is None or obj_class(held) != "knife":
            return {"violation": True, "category": "omitted_prerequisite", "blocker": "knife_1"}
    return {"violation": False, "category": "none", "blocker": None}

def _repair_for(verdict, trio):
    c, b = verdict["category"], verdict["blocker"]
    if c == "obstruction":
        return [{"action": "pick", "object": b, "target": None},
                {"action": "put", "object": b, "target": "clear_location"}]
    if c == "containment":
        return [{"action": "open", "object": b, "target": None}]
    if c == "occupancy":
        return [{"action": "put", "object": b, "target": "clear_location"}]
    if c == "omitted_prerequisite":
        if trio["action"] == "slice":
            return [{"action": "pick", "object": "knife_1", "target": None}]
        if trio["action"] == "put":
            return [{"action": "pick", "object": trio["object"], "target": None}]
    return []

def _apply(scene, trio, held):
    
    a, o, t = trio["action"], trio["object"], trio.get("target")
    on = [(s, tt) for s, tt in scene.on if s != o]
    if a == "pick":
        held = o
    elif a == "put":
        held = None
        if t and t != "clear_location":
            on.append((o, t))
    scene.on = on
    return held

def stage_verify_repair(scene, plan: List[Dict], max_iters=30) -> List[Dict]:
    banner("STAGE 3 — VERIFY + REPAIR (detect & insert missing steps)")
                                           
    work = SceneLite(objects=list(scene.objects), held=getattr(scene, "held", None),
                     open_objects=list(getattr(scene, "open_objects", [])),
                     on=list(scene.on), inside=list(getattr(scene, "inside", [])))
    held = work.held
    plan = list(plan)
    repaired: List[Dict] = []
    i = it = 0
    while i < len(plan) and it < max_iters:
        it += 1
        trio = plan[i]
        if USE_REAL_VERIFIER:
            verdict = _real_verify(work, trio, held)                 
        else:
            verdict = _rule_verify(work, trio, held)
        if verdict["violation"]:
            fix = _repair_for(verdict, trio)
            if fix:
                print(f"    ! {tuple(trio.values())}: {verdict['category']} "
                      f"(blocker={verdict['blocker']}) -> insert "
                      f"{[ (f['action'],f['object']) for f in fix ]}")
                plan[i:i] = fix
                continue
        repaired.append(trio)
        held = _apply(work, trio, held)
        i += 1
    print("  repaired plan:")
    for j, t in enumerate(repaired):
        print(f"    {j}: ({t['action']}, {t['object']}, {t.get('target')})")
    return repaired

def _real_verify(scene, trio, held):
    
    raise NotImplementedError("set USE_REAL_VERIFIER=False or implement _real_verify")


                                                                              
                                
                                                                              
def _real_sensetask(obj_id, task, image):
    
    raise NotImplementedError("set USE_REAL_SENSETASK=False or implement _real_sensetask")

def stage_grasp(repaired: List[Dict], image: Optional[str]) -> List[Dict]:
    banner("STAGE 4+5 — ROUTING + GRASPING (trio -> grasp pose)")
    out = []
    for trio in repaired:
        a = trio["action"]
        if a == "put" or a not in GRASP_ACTIONS:
            out.append({"trio": trio, "grasp": None, "route": "place/none"})
            print(f"    ({a}, {trio['object']}): no grasp (placement/non-grasp)")
            continue
        cat = obj_class(trio["object"])
        if cat in SENSETASK_OBJECTS:
            task = ACTION_TO_TASK.get(a, "Stabilizing")
            if USE_REAL_SENSETASK:
                grasp = _real_sensetask(trio["object"], task, image)
            else:
                grasp = f"SE2_GRASP(stub, task={task})"
            out.append({"trio": trio, "grasp": grasp, "route": f"SenseTask[{task}]"})
            print(f"    ({a}, {trio['object']}): SenseTask task='{task}' -> {grasp}")
        else:
            grasp = "GENERIC_GRASP(stub)" if not USE_REAL_SENSETASK else "GENERIC_GRASP"
            out.append({"trio": trio, "grasp": grasp, "route": "generic"})
            print(f"    ({a}, {trio['object']}): generic grasp -> {grasp}")
    return out


                                                                              
                            
                                                                              
def _real_execute(trio, grasp):
    
    raise NotImplementedError("set USE_REAL_ROBOT=False or implement _real_execute")

def stage_execute(grasp_plan: List[Dict]):
    banner("STAGE 6 — EXECUTION (trio + grasp -> arm motion)")
    for step in grasp_plan:
        trio, grasp = step["trio"], step["grasp"]
        if USE_REAL_ROBOT:
            _real_execute(trio, grasp)
        tag = f" with {grasp}" if grasp else ""
        print(f"    EXECUTE ({trio['action']}, {trio['object']}, {trio.get('target')}){tag}")
    if not USE_REAL_ROBOT:
        print("\n  [stub] no arm moved. Set USE_REAL_ROBOT=True and add depth to go physical.")


                                                                              
               
                                                                              
def run(command: str, image: Optional[str], classes: List[str]):
    print(f"\nLARP PIPELINE  |  command=\"{command}\"  image={image}")
    print("flags:",
          f"perception={USE_REAL_PERCEPTION} planner={USE_REAL_PLANNER}",
          f"verifier={USE_REAL_VERIFIER} sensetask={USE_REAL_SENSETASK}",
          f"robot={USE_REAL_ROBOT}")
    scene = stage_perception(image, classes)
    plan = stage_planning(command, scene)
    repaired = stage_verify_repair(scene, plan)
    grasp_plan = stage_grasp(repaired, image)
    stage_execute(grasp_plan)
    banner("PIPELINE COMPLETE")
    print(f"  steps planned (after repair): {len(repaired)}")
    print(f"  grasps synthesized: {sum(1 for g in grasp_plan if g['grasp'])}")
    return {"scene": scene, "repaired_plan": repaired, "grasp_plan": grasp_plan}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="make a sandwich")
    ap.add_argument("--image", default=None, help="table image (used if perception is real)")
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    args = ap.parse_args()
    run(args.command, args.image, args.classes)
