
import sys, types, random

                                                                              
fake_objects = [
    {"objectId":"Plate|1","objectType":"Plate","position":{"x":1.0,"y":0.9,"z":0.2},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Bread|1","objectType":"Bread","position":{"x":1.2,"y":0.9,"z":0.3},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Tomato|1","objectType":"Tomato","position":{"x":1.3,"y":0.9,"z":0.4},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Knife|1","objectType":"Knife","position":{"x":1.4,"y":0.9,"z":0.5},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Lettuce|1","objectType":"Lettuce","position":{"x":1.5,"y":0.9,"z":0.6},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Bowl|1","objectType":"Bowl","position":{"x":1.6,"y":0.9,"z":0.7},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Egg|1","objectType":"Egg","position":{"x":1.7,"y":0.9,"z":0.8},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Cheese|1","objectType":"Cheese","position":{"x":1.8,"y":0.9,"z":0.9},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["CounterTop|0"]},
    {"objectId":"Fridge|1","objectType":"Fridge","position":{"x":2.0,"y":1.0,"z":1.0},
     "isOpen":True,"isPickedUp":False,"isSliced":False,"openable":True,"parentReceptacles":None},
    {"objectId":"Milk|1","objectType":"Milk","position":{"x":2.1,"y":1.0,"z":1.1},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":["Fridge|1"]},
    {"objectId":"CounterTop|0","objectType":"CounterTop","position":{"x":1.0,"y":0.0,"z":0.0},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":None},
    {"objectId":"DiningTable|0","objectType":"DiningTable","position":{"x":0.0,"y":0.0,"z":0.0},
     "isOpen":False,"isPickedUp":False,"isSliced":False,"openable":False,"parentReceptacles":None},
]

class FakeEvent:
    def __init__(self): self.metadata={"objects":fake_objects,"lastActionSuccess":True}
    @property
    def frame(self):
        import numpy as np; return np.zeros((4,4,3),dtype="uint8")

class FakeController:
    def __init__(self,**kw): self.last_event=FakeEvent()
    def reset(self,scene=None): self.last_event=FakeEvent(); return self.last_event
    def step(self,**kw): self.last_event=FakeEvent(); return self.last_event
    def stop(self): pass

ai2thor=types.ModuleType("ai2thor")
ctrl_mod=types.ModuleType("ai2thor.controller"); ctrl_mod.Controller=FakeController
plat_mod=types.ModuleType("ai2thor.platform"); plat_mod.CloudRendering=object
ai2thor.controller=ctrl_mod; ai2thor.platform=plat_mod
sys.modules["ai2thor"]=ai2thor; sys.modules["ai2thor.controller"]=ctrl_mod; sys.modules["ai2thor.platform"]=plat_mod

                                                               
import generate
generate.generate(out_dir="./_thor_mock", mode="thor", n_per_task=3, seed=1, save_rgb=False)

                           
import json, glob
eps=glob.glob("./_thor_mock/episodes/*/*.json")
print(f"\nepisodes generated: {len(eps)}")
assert eps, "no episodes generated in thor mode"
                                                                
e=json.load(open([p for p in eps if 'complete' in p][0]))
n0=e["steps"][0]["state_before"]["nodes"][0]
has_pos=any("position" in n for n in e["steps"][0]["state_before"]["nodes"])
print("grounded nodes carry positions from metadata:", has_pos)
                                                             
viol=[s for s in e["steps"] if s["labels"]["violation"]]
print("violations in a COMPLETE thor episode:", len(viol), "(should be 0)")
                                  
import collections
cat=collections.Counter(json.loads(l)["corruption_type"] for l in open("./_thor_mock/manifest.jsonl"))
print("corruption categories in thor mode:", dict(cat))
print("\nMOCK THOR TEST: PASS" if has_pos and len(viol)==0 else "MOCK THOR TEST: CHECK")
