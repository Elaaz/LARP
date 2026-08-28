

from typing import Dict, List, Optional, Tuple
from domain import Trio
from worldstate import WorldState


                                                                             
def _normalize_type(t: str) -> str:
    
    mapping = {
        "Bread": "bread", "Tomato": "tomato", "Lettuce": "lettuce",
        "Apple": "apple", "Plate": "plate", "Bowl": "bowl", "Cup": "cup",
        "Mug": "cup", "Knife": "knife", "ButterKnife": "knife",
        "Fork": "fork", "Spoon": "spoon", "Fridge": "fridge",
        "Drawer": "drawer", "Cabinet": "cabinet", "CounterTop": "table",
        "DiningTable": "table", "Egg": "egg",
    }
    return mapping.get(t, t.lower())


def worldstate_from_metadata(objects: List[Dict],
                             id_map: Optional[Dict[str, str]] = None
                             ) -> Tuple[WorldState, Dict[str, str]]:
    
    s = WorldState()
    if id_map is None:
        id_map = {}
        counters: Dict[str, int] = {}
        for o in objects:
            cls = _normalize_type(o["objectType"])
            counters[cls] = counters.get(cls, 0) + 1
            id_map[o["objectId"]] = f"{cls}_{counters[cls]}"

               
    for o in objects:
        tid = o["objectId"]
        if tid not in id_map:
                                                                   
            cls = _normalize_type(o["objectType"])
            id_map[tid] = f"{cls}_x{len(id_map)}"
        oid = id_map[tid]
        s.add_object(oid, _normalize_type(o["objectType"]),
                     position=[o["position"]["x"], o["position"]["y"], o["position"]["z"]]
                     if "position" in o else None,
                     is_open=bool(o.get("isOpen", False)))
        if o.get("isPickedUp", False):
            s.set_held(oid, True)
        if o.get("isSliced", False):
            s.set_sliced(oid, True)

                                      
    for o in objects:
        tid = o["objectId"]
        oid = id_map[tid]
        parents = o.get("parentReceptacles") or []
        for p in parents:
            if p not in id_map:
                continue
            parent_id = id_map[p]
                                                                                 
            parent_obj = next((x for x in objects if x["objectId"] == p), None)
            if parent_obj and parent_obj.get("openable", False):
                s.add_in(oid, parent_id)
            else:
                s.add_on(oid, parent_id)

    return s, id_map


                                                                              
def trio_to_thor_action(trio: Trio, id_map_rev: Dict[str, str]) -> Optional[Dict]:
    
    obj_thor = id_map_rev.get(trio.object)
    tgt_thor = id_map_rev.get(trio.target) if trio.target else None

    if trio.action == "pick":
        return {"action": "PickupObject", "objectId": obj_thor}
    if trio.action == "put":
                                                                                 
        return {"action": "PutObject", "objectId": tgt_thor} if tgt_thor else None
    if trio.action == "open":
        return {"action": "OpenObject", "objectId": obj_thor}
    if trio.action == "close":
        return {"action": "CloseObject", "objectId": obj_thor}
    if trio.action == "slice":
        return {"action": "SliceObject", "objectId": obj_thor}
    return None


class ThorSession:
    

    def __init__(self, scene: str = "FloorPlan1", width: int = 600,
                 height: int = 600, headless: bool = True):
        try:
            from ai2thor.controller import Controller
        except Exception as e:
            raise RuntimeError(
                "ai2thor is not installed/available here. Install with "
                "`pip install ai2thor` on a machine with rendering. "
                f"Original error: {e}")
        kwargs = dict(scene=scene, width=width, height=height,
                      renderInstanceSegmentation=False)
        if headless:
                                                                               
            try:
                from ai2thor.platform import CloudRendering
                kwargs["platform"] = CloudRendering
            except Exception:
                pass
        self.controller = Controller(**kwargs)
        self.last_event = self.controller.last_event

    def reset(self, scene: str):
        self.last_event = self.controller.reset(scene=scene)
        return self.last_event

    def objects(self) -> List[Dict]:
        return self.last_event.metadata["objects"]

    def step(self, **kwargs):
        self.last_event = self.controller.step(**kwargs)
        return self.last_event

    def last_success(self) -> bool:
        return bool(self.last_event.metadata.get("lastActionSuccess", False))

    def rgb(self):
        return self.last_event.frame               

    def stop(self):
        try:
            self.controller.stop()
        except Exception:
            pass


                                                                               
def resolve_required_types(objects: List[Dict],
                           required_types: List[str],
                           rng) -> Optional[Dict[str, str]]:
    
                                                          
    by_class: Dict[str, List[str]] = {}
    for o in objects:
        c = _normalize_type(o["objectType"])
        by_class.setdefault(c, []).append(o["objectId"])
    for v in by_class.values():
        rng.shuffle(v)

    chosen: Dict[str, str] = {}
    used: set = set()
    for req in required_types:
        cls = req.split("#")[0]
        pool = [oid for oid in by_class.get(cls, []) if oid not in used]
        if not pool:
            return None                                           
        oid = pool[0]
        used.add(oid)
        chosen[req] = oid
    return chosen


def capture_rgb_png(session: "ThorSession", path: str):
    
    try:
        from PIL import Image
        Image.fromarray(session.rgb()).save(path)
    except Exception:
        pass


                                                                             
                                             
                                                                        
                                            
                                          
                                                         
                                                                             
