

import random
from domain import Trio
from worldstate import WorldState
from labeler import roll_forward, CORRUPTORS


def base_scene():
    
    s = WorldState()
    s.add_object("table_1", "table")
    s.add_object("plate_1", "plate")
    s.add_object("bottom_bread_1", "bottom_bread")
    s.add_object("meat_1", "meat")
    s.add_object("top_bread_1", "top_bread")
    s.add_object("tomato_1", "tomato")
    s.add_object("knife_1", "knife")
    for o in ["plate_1", "bottom_bread_1", "meat_1", "top_bread_1", "tomato_1", "knife_1"]:
        s.add_on(o, "table_1")
    return s


def sandwich_plan():
    return [
        Trio("pick", "bottom_bread_1"),
        Trio("put", "bottom_bread_1", "plate_1"),
        Trio("pick", "meat_1"),
        Trio("put", "meat_1", "bottom_bread_1"),
        Trio("pick", "top_bread_1"),
        Trio("put", "top_bread_1", "meat_1"),
    ]


PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  PASS  {name}")
    else:
        FAIL += 1; print(f"  FAIL  {name}")


print("=== Test 1: complete plan has NO violations ===")
labels = roll_forward(base_scene(), sandwich_plan())
check("all steps executable", all(not l.violation for l in labels))
check("six steps labeled", len(labels) == 6)
                                                                      
step1_after = labels[1].state_after
plate_node = next(n for n in step1_after["nodes"] if n["id"] == "plate_1")
check("plate not clear after putting bread on it", plate_node["state"]["clear"] is False)


print("\n=== Test 2: omitted prerequisite (remove a pick) ===")
rng = random.Random(0)
                                                               
plan = sandwich_plan()
corrupted_plan = plan[1:]                                                                               
labels = roll_forward(base_scene(), corrupted_plan)
first = labels[0]
check("first step is a violation", first.violation)
check("category is omitted_prerequisite", first.failure_category == "omitted_prerequisite")
check("blocking relation is held", first.blocking_relation == "held")
check("gold repair re-inserts the pick",
      first.gold_repair and first.gold_repair[0]["action"] == "pick"
      and first.gold_repair[0]["object"] == "bottom_bread_1")


print("\n=== Test 3: obstruction (apple on plate the plan picks) ===")
s = base_scene()
                                  
plan = [Trio("pick", "plate_1"), Trio("put", "plate_1", "table_1")]
out = CORRUPTORS["obstruction"](s, plan, random.Random(1))
check("obstruction corruptor fired", out is not None)
cs, cplan, meta = out
labels = roll_forward(cs, cplan)
check("pick plate is blocked", labels[0].violation)
check("category obstruction", labels[0].failure_category == "obstruction")
check("blocking relation on", labels[0].blocking_relation == "on")
check("blocking object is the distractor", labels[0].blocking_object == meta["distractor"])
check("repair clears the blocker first",
      labels[0].gold_repair[0]["action"] == "pick"
      and labels[0].gold_repair[0]["object"] == meta["distractor"])


print("\n=== Test 4: containment (object inside closed fridge) ===")
s = base_scene()
s.add_object("fridge_1", "fridge", is_open=False)
s.add_object("milk_1", "milk")
plan = [Trio("pick", "milk_1"), Trio("put", "milk_1", "table_1")]
                                     
s.add_in("milk_1", "fridge_1")
labels = roll_forward(s, plan)
check("pick milk blocked by containment", labels[0].violation and labels[0].failure_category == "containment")
check("blocking object is fridge", labels[0].blocking_object == "fridge_1")
check("repair opens the container", labels[0].gold_repair[0]["action"] == "open"
      and labels[0].gold_repair[0]["object"] == "fridge_1")


print("\n=== Test 5: occupancy (gripper already holding something) ===")
s = base_scene()
out = CORRUPTORS["occupancy"](s, sandwich_plan(), random.Random(2))
check("occupancy corruptor fired", out is not None)
cs, cplan, meta = out
labels = roll_forward(cs, cplan)
check("first pick blocked by occupancy", labels[0].violation and labels[0].failure_category == "occupancy")
check("blocking relation held", labels[0].blocking_relation == "held")
check("repair puts down the held object", labels[0].gold_repair[0]["action"] == "put")


print("\n=== Test 6: slice requires knife (benchmark-defined) ===")
s = base_scene()
                                                      
plan = [Trio("slice", "tomato_1")]
labels = roll_forward(s, plan)
check("slice without knife blocked", labels[0].violation)
check("category omitted_prerequisite", labels[0].failure_category == "omitted_prerequisite")
check("repair is pick knife", labels[0].gold_repair[0]["object"] == "knife")
                           
plan2 = [Trio("pick", "knife_1"), Trio("slice", "tomato_1")]
labels2 = roll_forward(s, plan2)
check("slice with knife held succeeds", not labels2[1].violation)


print("\n=== Test 7: ordering corruption ===")
out = CORRUPTORS["ordering"](base_scene(), sandwich_plan(), random.Random(3))
check("ordering corruptor fired", out is not None)
if out:
    cs, cplan, meta = out
    labels = roll_forward(cs, cplan)
    check("a put-before-its-pick produces a violation", any(l.violation for l in labels))


print(f"\n================  {PASS} passed, {FAIL} failed  ================")
exit(1 if FAIL else 0)
