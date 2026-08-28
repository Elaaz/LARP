
from larp_agent import Trio, ITG, Scene, validate_itg

P, F = 0, 0
def check(name, cond):
    global P, F
    print(("  PASS " if cond else "  FAIL ") + name); P += cond; F += (not cond)

def chain(trios): return ITG(trios=trios, edges=[(i,i+1) for i in range(len(trios)-1)])

scene = Scene(
    objects=["plate_1","bottom_bread_1","meat_1","tomato_1","top_bread_1","knife_1","table_1","fridge_1","milk_1"],
    on=[("bottom_bread_1","table_1"),("meat_1","table_1"),("tomato_1","table_1"),
        ("top_bread_1","table_1"),("knife_1","table_1"),("plate_1","table_1")],
    inside=[("milk_1","fridge_1")], open_objects=[],
)

print("=== valid sandwich plan ===")
good = chain([
    Trio("pick","bottom_bread_1"), Trio("put","bottom_bread_1","plate_1"),
    Trio("pick","meat_1"), Trio("put","meat_1","bottom_bread_1"),
    Trio("pick","top_bread_1"), Trio("put","top_bread_1","meat_1"),
])
errs = validate_itg(good, scene)
check("valid plan has no errors", errs == [])

print("=== two objects at once (2-finger violation) ===")
bad = chain([Trio("pick","meat_1"), Trio("pick","tomato_1")])
errs = validate_itg(bad, scene)
check("catches holding two objects", any("one object at a time" in e for e in errs))

print("=== pick an ungraspable big object (table) ===")
errs = validate_itg(chain([Trio("pick","table_1")]), scene)
check("rejects picking the table", any("not graspable" in e or "exceeds" in e for e in errs))

print("=== object not in scene ===")
errs = validate_itg(chain([Trio("pick","banana_1")]), scene)
check("rejects unknown object", any("not in scene" in e for e in errs))

print("=== unknown action ===")
errs = validate_itg(chain([Trio("teleport","meat_1")]), scene)
check("rejects unknown action", any("unknown action" in e for e in errs))

print("=== slice without holding knife ===")
errs = validate_itg(chain([Trio("slice","tomato_1")]), scene)
check("rejects slice with empty gripper", any("needs a tool" in e for e in errs))

print("=== slice WITH knife held first (valid) ===")
errs = validate_itg(chain([Trio("pick","knife_1"), Trio("slice","tomato_1")]), scene)
check("valid slice after picking knife", errs == [])

print("=== put without holding (omitted pick) ===")
errs = validate_itg(chain([Trio("put","meat_1","plate_1")]), scene)
check("rejects put with nothing held", any("must be holding" in e for e in errs))

print("=== pick milk inside closed fridge ===")
errs = validate_itg(chain([Trio("pick","milk_1")]), scene)
check("rejects pick from closed container", any("closed" in e for e in errs))

print("=== open fridge then pick milk (valid) ===")
errs = validate_itg(chain([Trio("open","fridge_1"), Trio("pick","milk_1")]), scene)
check("valid after opening fridge", errs == [])

print("=== pick a blocked (not clear) object ===")
blocked_scene = Scene(objects=["plate_1","apple_1","table_1"],
                      on=[("apple_1","plate_1"),("plate_1","table_1")])
errs = validate_itg(chain([Trio("pick","plate_1")]), blocked_scene)
check("rejects picking obstructed object", any("not clear" in e for e in errs))

print(f"\n{P} passed, {F} failed")
exit(1 if F else 0)
