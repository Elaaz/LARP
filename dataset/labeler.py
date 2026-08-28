

import random
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from domain import (Trio, PRECONDITIONS, EFFECTS, Violation, FAILURE_CATEGORIES)
from worldstate import WorldState


                                                                             
                       
                                                                             

@dataclass
class StepLabel:
    step_index: int
    trio: Trio
    state_before: Dict                                
    state_after: Dict                                                                        
    violation: bool
    failure_category: Optional[str]
    blocking_object: Optional[str]
    blocking_relation: Optional[str]
    gold_repair: List[Dict]                                                    


def repair_for(violation: Violation, trio: Trio) -> List[Trio]:
    
    cat = violation.category
    if cat == "obstruction":
        blk = violation.blocking_object
        return [Trio("pick", blk, None), Trio("put", blk, "clear_location")]
    if cat == "containment":
        return [Trio("open", violation.blocking_object, None)]
    if cat == "occupancy":
        blk = violation.blocking_object
        return [Trio("put", blk, "clear_location")]
    if cat == "omitted_prerequisite":
                                                               
        if trio.action == "slice":
            return [Trio("pick", "knife", None)]
        if trio.action == "put":
                                                
            return [Trio("pick", trio.object, None)]
    return []                                                


def roll_forward(initial: WorldState, plan: List[Trio],
                 stop_at_first_violation: bool = False) -> List[StepLabel]:
    
    state = initial.clone()
    labels: List[StepLabel] = []

    for i, trio in enumerate(plan):
        before = state.to_graph()
        pre_fn = PRECONDITIONS.get(trio.action)
        violation = pre_fn(state, trio) if pre_fn else None

        if violation is None:
                          
            EFFECTS[trio.action](state, trio)
            after = state.to_graph()
            labels.append(StepLabel(
                step_index=i, trio=trio, state_before=before, state_after=after,
                violation=False, failure_category=None,
                blocking_object=None, blocking_relation=None, gold_repair=[]))
        else:
            after = before                                              
            repair = repair_for(violation, trio)
            labels.append(StepLabel(
                step_index=i, trio=trio, state_before=before, state_after=after,
                violation=True, failure_category=violation.category,
                blocking_object=violation.blocking_object,
                blocking_relation=violation.blocking_relation,
                gold_repair=[t.as_dict() for t in repair]))
            if stop_at_first_violation:
                break

    return labels


                                                                             
                       
                                                                             
                                                                             
                                  

def corrupt_omitted_prerequisite(initial: WorldState, plan: List[Trio],
                                 rng: random.Random) -> Optional[Tuple]:
    
    candidates = [i for i, t in enumerate(plan) if t.action in ("pick", "open")]
    if not candidates:
        return None
    idx = rng.choice(candidates)
    new_plan = plan[:idx] + plan[idx + 1:]
    meta = {"corruption_type": "omitted_prerequisite",
            "removed_index": idx, "removed_trio": plan[idx].as_dict()}
    return initial.clone(), new_plan, meta


def corrupt_ordering(initial: WorldState, plan: List[Trio],
                     rng: random.Random) -> Optional[Tuple]:
    
                                                            
    for i, t in enumerate(plan):
        if t.action == "pick":
            for j in range(i + 1, len(plan)):
                if plan[j].action == "put" and plan[j].object == t.object:
                    new_plan = list(plan)
                    new_plan[i], new_plan[j] = new_plan[j], new_plan[i]
                    meta = {"corruption_type": "ordering",
                            "swapped": [i, j]}
                    return initial.clone(), new_plan, meta
    return None


def corrupt_obstruction(initial: WorldState, plan: List[Trio],
                        rng: random.Random) -> Optional[Tuple]:
    
    picked_receptacles = [t.object for t in plan
                          if t.action == "pick" and initial.exists(t.object)
                          and "container" in initial._aff(t.object)]
    if not picked_receptacles:
                                                              
        picked_receptacles = [t.object for t in plan if t.action == "pick"]
        if not picked_receptacles:
            return None
    target = rng.choice(picked_receptacles)
    s = initial.clone()
                                                  
    distractor = "distractor_apple_1"
    s.add_object(distractor, "apple")
    s.add_on(distractor, target)
    meta = {"corruption_type": "obstruction",
            "distractor": distractor, "on_target": target}
    return s, list(plan), meta


def corrupt_containment(initial: WorldState, plan: List[Trio],
                        rng: random.Random) -> Optional[Tuple]:
    
    picked = [t.object for t in plan if t.action == "pick" and initial.exists(t.object)]
    containers = [oid for oid, c in initial.cls.items()
                  if "openable" in initial._aff(oid)]
    if not picked or not containers:
        return None
    obj = rng.choice(picked)
    cont = rng.choice(containers)
    s = initial.clone()
    s.set_open(cont, False)                         
    s.add_in(obj, cont)                                     
    meta = {"corruption_type": "containment", "object": obj, "container": cont}
    return s, list(plan), meta


def corrupt_occupancy(initial: WorldState, plan: List[Trio],
                      rng: random.Random) -> Optional[Tuple]:
    
    if not any(t.action == "pick" for t in plan):
        return None
    s = initial.clone()
    distractor = "distractor_held_1"
    s.add_object(distractor, "cup")
    s.set_held(distractor, True)
    meta = {"corruption_type": "occupancy", "held": distractor}
    return s, list(plan), meta


CORRUPTORS = {
    "omitted_prerequisite": corrupt_omitted_prerequisite,
    "ordering": corrupt_ordering,
    "obstruction": corrupt_obstruction,
    "containment": corrupt_containment,
    "occupancy": corrupt_occupancy,
}
