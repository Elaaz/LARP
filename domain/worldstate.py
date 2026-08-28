

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import copy

from domain import affordances_of


class WorldState:
    def __init__(self):
                                     
        self.cls: Dict[str, str] = {}
                                                                                  
        self.pos: Dict[str, List[float]] = {}
                          
        self.flag_held: Set[str] = set()
        self.flag_open: Set[str] = set()
        self.flag_sliced: Set[str] = set()
               
        self._on: Set[Tuple[str, str]] = set()                   
        self._in: Set[Tuple[str, str]] = set()                    

                                                                             
    def add_object(self, inst_id: str, object_class: str,
                   position: Optional[List[float]] = None,
                   is_open: bool = False):
        self.cls[inst_id] = object_class
        if position is not None:
            self.pos[inst_id] = list(position)
        if is_open:
            self.flag_open.add(inst_id)

    def add_on(self, a: str, b: str):
        self._on.add((a, b))

    def add_in(self, a: str, b: str):
        self._in.add((a, b))

                                                                            
    def _aff(self, inst_id: str) -> List[str]:
        return affordances_of(self.cls.get(inst_id, ""))

    def exists(self, inst_id: str) -> bool:
        return inst_id in self.cls

    def is_container(self, inst_id: str) -> bool:
        return "container" in self._aff(inst_id)

    def is_openable(self, inst_id: str) -> bool:
        return "openable" in self._aff(inst_id)

    def is_tool(self, inst_id: str) -> bool:
        return "tool" in self._aff(inst_id)

                                                                             
    def is_held(self, inst_id: str) -> bool:
        return inst_id in self.flag_held

    def held_object(self) -> Optional[str]:
                                                            
        return next(iter(self.flag_held), None)

    def is_open(self, inst_id: str) -> bool:
        return inst_id in self.flag_open

    def is_sliced(self, inst_id: str) -> bool:
        return inst_id in self.flag_sliced

    def objects_on(self, inst_id: str) -> List[str]:
        
        return sorted([a for (a, b) in self._on if b == inst_id])

    def container_of(self, inst_id: str) -> Optional[str]:
        
        for (a, b) in self._in:
            if a == inst_id:
                return b
        return None

    def is_clear(self, inst_id: str) -> bool:
        return len(self.objects_on(inst_id)) == 0

                                                                             
    def set_held(self, inst_id: str, value: bool):
        if value:
            self.flag_held.add(inst_id)
        else:
            self.flag_held.discard(inst_id)

    def set_open(self, inst_id: str, value: bool):
        if value:
            self.flag_open.add(inst_id)
        else:
            self.flag_open.discard(inst_id)

    def set_sliced(self, inst_id: str, value: bool):
        if value:
            self.flag_sliced.add(inst_id)

    def remove_support_edges_to(self, inst_id: str):
        self._on = {(a, b) for (a, b) in self._on if a != inst_id}

    def remove_containment_edges_to(self, inst_id: str):
        self._in = {(a, b) for (a, b) in self._in if a != inst_id}

                                                                             
    def clone(self) -> "WorldState":
        return copy.deepcopy(self)

    def to_graph(self) -> Dict:
        
        nodes = []
        for inst_id, c in sorted(self.cls.items()):
            node = {
                "id": inst_id,
                "type": c,
                "state": {
                    "held": self.is_held(inst_id),
                    "open": self.is_open(inst_id) if self.is_openable(inst_id) else None,
                    "sliced": self.is_sliced(inst_id),
                    "clear": self.is_clear(inst_id),
                },
            }
            if inst_id in self.pos:
                node["position"] = self.pos[inst_id]
            nodes.append(node)

        edges = []
        for (a, b) in sorted(self._on):
            edges.append({"subject": a, "relation": "on", "object": b})
        for (a, b) in sorted(self._in):
            edges.append({"subject": a, "relation": "in", "object": b})
        return {"nodes": nodes, "edges": edges}
