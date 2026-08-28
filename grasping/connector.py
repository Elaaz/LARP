

from typing import Dict, List, Optional, Callable

                                                                             
                                                                      
                                                                             
                                                                             
ACTION_TO_TASK: Dict[str, str] = {
    "slice": "cut",                 
    "pour":  "pour",                                                      
    "open":  "open",                     
    "close": "close",                                                            
    "pick":  "hold",                          
}

                                                                             
                                                                         
                                                                           
                            
                                                                             
SENSETASK_OBJECTS = {"knife", "cup"}                                           

def object_category(obj_id: str) -> str:
    
    return obj_id.rsplit("_", 1)[0]

                                                                             
                                                    
                                                                             
                                                 
                                                                    
                                                                              
                                                                             
GRASP_ACTIONS = {"pick", "slice", "open", "close", "pour"}

def run_plan(repaired_plan: List[Dict],
             resolve: Callable, sensetask: Callable,
             generic_grasp: Callable, execute: Callable,
             llm_fallback: Optional[Callable] = None):
    
    for trio in repaired_plan:
        action = trio["action"]

        if action == "put":
            execute(trio, grasp=None)                                         
            continue

        if action not in GRASP_ACTIONS:
            execute(trio, grasp=None)                                          
            continue

        inst = resolve(trio["object"])                                          
        cat = object_category(trio["object"])

        if cat in SENSETASK_OBJECTS:
            task = ACTION_TO_TASK.get(action)
            if task is None and llm_fallback is not None:
                task = llm_fallback(trio)                                       
            grasp = sensetask(inst, task)                                 
        else:
            grasp = generic_grasp(inst)                                       

        execute(trio, grasp=grasp)
