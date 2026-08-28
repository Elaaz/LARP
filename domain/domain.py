

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple

                                                                             
              
                                                                             

                                                                           
                                                 
AFFORDANCES = ["graspable", "stackable", "cuttable", "container", "openable",
               "surface", "tool"]

                                       
STATE_KEYS = ["held", "open", "sliced", "clear"]

                                    
RELATIONS = ["on", "in", "supports", "contains"]
                                                      
                                      
                                      

                                                                          
ACTIONS = ["pick", "put", "open", "close", "slice"]

                                                       
FAILURE_CATEGORIES = [
    "obstruction",                                                       
    "containment",                                                         
    "occupancy",                                                            
    "omitted_prerequisite",                                                   
    "ordering",                                                       
]


                                                                             
      
                                                                             

@dataclass(frozen=True)
class Trio:
    
    action: str
    object: str
    target: Optional[str] = None

    def as_dict(self) -> Dict:
        return {"action": self.action, "object": self.object, "target": self.target}

    @staticmethod
    def from_dict(d: Dict) -> "Trio":
        return Trio(d["action"], d["object"], d.get("target"))

    def __repr__(self):
        t = self.target if self.target is not None else "\u2205"
        return f"({self.action}, {self.object}, {t})"


                                                                             
                                    
                                                                             

@dataclass
class Violation:
    
    category: str                                                                
    blocking_object: Optional[str]                                    
    blocking_relation: Optional[str]                            

    def as_dict(self) -> Dict:
        return {
            "category": self.category,
            "blocking_object": self.blocking_object,
            "blocking_relation": self.blocking_relation,
        }


                                                               
                                                         
                                                                         


                                                                             
               
                                                                             

def pre_pick(state, trio: "Trio") -> Optional[Violation]:
    obj = trio.object
    if not state.exists(obj):
        return Violation("unknown", obj, None)
                                       
    held = state.held_object()
    if held is not None:
        return Violation("occupancy", held, "held")
                                                                           
    container = state.container_of(obj)
    if container is not None and not state.is_open(container):
        return Violation("containment", container, "in")
                                                                        
                                                                              
                                               
    supported = state.objects_on(obj)
    if supported:
        return Violation("obstruction", supported[0], "on")
    return None


def pre_put(state, trio: "Trio") -> Optional[Violation]:
    obj, tgt = trio.object, trio.target
    if not state.exists(obj):
        return Violation("unknown", obj, None)
                                                                        
    if not state.is_held(obj):
        return Violation("omitted_prerequisite", obj, "held")
    if tgt is not None and tgt != "clear_location":
        if not state.exists(tgt):
            return Violation("unknown", tgt, None)
                                                                             
        if state.is_container(tgt) and state.is_openable(tgt) and not state.is_open(tgt):
            return Violation("containment", tgt, "in")
    return None


def pre_open(state, trio: "Trio") -> Optional[Violation]:
    obj = trio.object
    if not state.exists(obj):
        return Violation("unknown", obj, None)
    if not state.is_openable(obj):
        return Violation("unknown", obj, None)
                                                                            
    held = state.held_object()
    if held is not None:
        return Violation("occupancy", held, "held")
    return None


def pre_close(state, trio: "Trio") -> Optional[Violation]:
    obj = trio.object
    if not state.exists(obj) or not state.is_openable(obj):
        return Violation("unknown", obj, None)
    return None


def pre_slice(state, trio: "Trio") -> Optional[Violation]:
    obj = trio.object
    if not state.exists(obj):
        return Violation("unknown", obj, None)
                                                                      
                                                                           
    held = state.held_object()
    if held is None or not state.is_tool(held):
                                               
        return Violation("omitted_prerequisite", "knife", "held")
    return None


PRECONDITIONS: Dict[str, Callable] = {
    "pick": pre_pick,
    "put": pre_put,
    "open": pre_open,
    "close": pre_close,
    "slice": pre_slice,
}


                                                                             
                                                                        
                                                                             

def eff_pick(state, trio: "Trio") -> None:
    state.set_held(trio.object, True)
    state.remove_support_edges_to(trio.object)                                  
    state.remove_containment_edges_to(trio.object)

def eff_put(state, trio: "Trio") -> None:
    state.set_held(trio.object, False)
    tgt = trio.target
    if tgt is None or tgt == "clear_location":
        return
    if state.is_container(tgt) and state.is_open(tgt):
        state.add_in(trio.object, tgt)
    else:
        state.add_on(trio.object, tgt)

def eff_open(state, trio: "Trio") -> None:
    state.set_open(trio.object, True)

def eff_close(state, trio: "Trio") -> None:
    state.set_open(trio.object, False)

def eff_slice(state, trio: "Trio") -> None:
    state.set_sliced(trio.object, True)


EFFECTS: Dict[str, Callable] = {
    "pick": eff_pick,
    "put": eff_put,
    "open": eff_open,
    "close": eff_close,
    "slice": eff_slice,
}


                                                                             
                                                                  
                                                                          
                                                                             

OBJECT_AFFORDANCES: Dict[str, List[str]] = {
    "bread":   ["graspable", "stackable", "cuttable"],
    "bottom_bread": ["graspable", "stackable", "cuttable"],
    "top_bread":    ["graspable", "stackable", "cuttable"],
    "meat":    ["graspable", "stackable"],
    "lettuce": ["graspable", "stackable", "cuttable"],
    "tomato":  ["graspable", "stackable", "cuttable"],
    "cheese":  ["graspable", "stackable", "cuttable"],
    "apple":   ["graspable", "stackable", "cuttable"],
    "plate":   ["graspable", "surface", "container"],
    "bowl":    ["graspable", "container"],
    "cup":     ["graspable", "container"],
    "knife":   ["graspable", "tool"],
    "fork":    ["graspable", "tool"],
    "spoon":   ["graspable", "tool"],
    "fridge":  ["openable", "container", "surface"],
    "drawer":  ["openable", "container"],
    "cabinet": ["openable", "container"],
    "table":   ["surface"],
    "milk":    ["graspable", "container"],
}


def affordances_of(object_class: str) -> List[str]:
    return OBJECT_AFFORDANCES.get(object_class, ["graspable"])
