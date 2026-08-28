

import os
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

                                                                             
                                                                     
                                                                             
ACTIONS = ["pick", "put", "open", "close", "slice"]

OBJECT_AFFORDANCES: Dict[str, List[str]] = {
    "bread": ["graspable", "stackable", "cuttable"],
    "bottom_bread": ["graspable", "stackable", "cuttable"],
    "top_bread": ["graspable", "stackable", "cuttable"],
    "meat": ["graspable", "stackable"],
    "lettuce": ["graspable", "stackable", "cuttable"],
    "tomato": ["graspable", "stackable", "cuttable"],
    "cheese": ["graspable", "stackable", "cuttable"],
    "apple": ["graspable", "stackable", "cuttable"],
    "plate": ["graspable", "surface", "container"],
    "bowl": ["graspable", "container"],
    "cup": ["graspable", "container"],
    "knife": ["graspable", "tool"],
    "fork": ["graspable", "tool"],
    "fridge": ["openable", "container", "surface"],
    "drawer": ["openable", "container"],
    "cabinet": ["openable", "container"],
    "table": ["surface"],
    "milk": ["graspable", "container"],
}

GRIPPER_MAX_WIDTH_M = 0.085
OBJECT_WIDTH_M: Dict[str, float] = {
    "bread": 0.06, "bottom_bread": 0.06, "top_bread": 0.06, "meat": 0.05,
    "lettuce": 0.07, "tomato": 0.05, "cheese": 0.04, "apple": 0.07,
    "plate": 0.05, "bowl": 0.06, "cup": 0.05, "knife": 0.02, "fork": 0.02,
    "milk": 0.07,
    "fridge": 9.99, "drawer": 9.99, "cabinet": 9.99, "table": 9.99,
}


def affordances_of(cls: str) -> List[str]:
    return OBJECT_AFFORDANCES.get(cls, ["graspable"])

def object_class(obj_id: str) -> str:
    return re.sub(r"_\d+$", "", obj_id)

def graspable_by_two_fingers(obj_id: str) -> Tuple[bool, str]:
    cls = object_class(obj_id)
    if "graspable" not in affordances_of(cls):
        return False, f"{obj_id} is not graspable (no graspable affordance)"
    w = OBJECT_WIDTH_M.get(cls, 0.05)
    if w > GRIPPER_MAX_WIDTH_M:
        return False, f"{obj_id} width {w:.3f}m exceeds 2-finger jaw {GRIPPER_MAX_WIDTH_M:.3f}m"
    return True, ""


                                                                             
                                         
                                                                             
@dataclass
class Trio:
    action: str
    object: str
    target: Optional[str] = None
    def tuple(self): return (self.action, self.object, self.target)

@dataclass
class ITG:
    trios: List[Trio] = field(default_factory=list)
    edges: List[Tuple[int, int]] = field(default_factory=list)
    def as_dict(self):
        return {"trios": [asdict(t) for t in self.trios], "edges": self.edges}

@dataclass
class Scene:
    objects: List[str]
    held: Optional[str] = None
    open_objects: List[str] = field(default_factory=list)
    on: List[Tuple[str, str]] = field(default_factory=list)
    inside: List[Tuple[str, str]] = field(default_factory=list)
    def exists(self, oid): return oid in self.objects
    def objects_on(self, oid): return [a for a, b in self.on if b == oid]
    def container_of(self, oid):
        return next((b for a, b in self.inside if a == oid), None)


                                                                             
                                                                  
                                                                             
def validate_itg(itg: ITG, scene: Scene) -> List[str]:
    errors: List[str] = []
    held = scene.held
    open_set = set(scene.open_objects)
    on = list(scene.on)
    inside = list(scene.inside)

    def objects_on(o): return [a for a, b in on if b == o]
    def container_of(o): return next((b for a, b in inside if a == o), None)

    for i, t in enumerate(itg.trios):
        tag = f"trio {i} {t.tuple()}"
        if t.action not in ACTIONS:
            errors.append(f"{tag}: unknown action '{t.action}'"); continue
        if not scene.exists(t.object):
            errors.append(f"{tag}: object '{t.object}' not in scene"); continue
        if t.target and t.target != "clear_location" and not scene.exists(t.target):
            errors.append(f"{tag}: target '{t.target}' not in scene"); continue

        if t.action == "pick":
            if held is not None:
                errors.append(f"{tag}: gripper already holding {held} (2-finger: one object at a time)")
            ok, why = graspable_by_two_fingers(t.object)
            if not ok:
                errors.append(f"{tag}: {why}")
            c = container_of(t.object)
            if c and c not in open_set:
                errors.append(f"{tag}: {t.object} is inside closed {c} (open it first)")
            blockers = objects_on(t.object)
            if blockers:
                errors.append(f"{tag}: {t.object} not clear, {blockers[0]} on top (clear it first)")
            if all(not e.startswith(tag) for e in errors):
                held = t.object
                on = [(a, b) for a, b in on if a != t.object]
                inside = [(a, b) for a, b in inside if a != t.object]
        elif t.action == "put":
            if held != t.object:
                errors.append(f"{tag}: must be holding {t.object} to put it (currently holding {held})")
            else:
                held = None
                if t.target and t.target != "clear_location":
                    on.append((t.object, t.target))
        elif t.action == "open":
            if "openable" not in affordances_of(object_class(t.object)):
                errors.append(f"{tag}: {t.object} is not openable")
            elif held is not None:
                errors.append(f"{tag}: need a free gripper to open (holding {held})")
            else:
                open_set.add(t.object)
        elif t.action == "close":
            if "openable" not in affordances_of(object_class(t.object)):
                errors.append(f"{tag}: {t.object} is not openable")
            else:
                open_set.discard(t.object)
        elif t.action == "slice":
            if held is None or "tool" not in affordances_of(object_class(held)):
                errors.append(f"{tag}: slicing needs a tool (knife) held first")
            if "cuttable" not in affordances_of(object_class(t.object)):
                errors.append(f"{tag}: {t.object} is not cuttable")
    return errors


                                                                             
                                             
                                                                             
SYSTEM_PROMPT = """You are the strategic planner for a kitchen robot with a TWO-FINGER parallel-jaw gripper.
Convert the user's command into an Instructional Task Graph: an ordered list of (action, object, target) trios.

HARD RULES — follow exactly:
- Allowed actions ONLY: pick, put, open, close, slice.
- Every object and target MUST be one of the objects listed in SCENE (use the exact ids). The only allowed non-scene target is "clear_location".
- Trio meaning: (action, object, target). For pick/open/close/slice, target is null.
  Examples: ("pick","knife_1",null), ("put","bread_1","plate_1"), ("slice","tomato_1",null), ("open","fridge_1",null).
- TWO-FINGER GRIPPER physical limits:
  * Hold ONE object at a time. To pick something new, put down what you hold first.
  * Only pick graspable objects that fit the jaw (no picking tables, fridges, or oversized items).
  * To pick an object, it must be clear (nothing on top) — if blocked, pick and move the blocker to clear_location first.
  * To pick an object inside a closed container, OPEN the container first.
  * To SLICE, you must be holding a knife first.
  * To PUT an object, you must currently be holding that exact object.

OUTPUT FORMAT — return ONLY valid JSON, no prose, no markdown fences:
{"trios":[{"action":"pick","object":"knife_1","target":null}, ...]}
"""

def build_user_prompt(command: str, scene: Scene, prior_errors: Optional[List[str]] = None) -> str:
    lines = [f'COMMAND: "{command}"', "", "SCENE objects:"]
    for o in scene.objects:
        cls = object_class(o)
        aff = ",".join(affordances_of(cls))
        lines.append(f"  - {o} (class={cls}; affordances={aff})")
    if scene.held: lines.append(f"Gripper currently holding: {scene.held}")
    if scene.open_objects: lines.append(f"Open: {scene.open_objects}")
    if scene.on: lines.append(f"On relations: {scene.on}")
    if scene.inside: lines.append(f"Inside relations: {scene.inside}")
    if prior_errors:
        lines += ["", "Your previous plan had these errors — fix them:"]
        lines += [f"  - {e}" for e in prior_errors]
    lines += ["", "Return ONLY the JSON ITG."]
    return "\n".join(lines)


def parse_itg(text: str) -> ITG:
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object found in model output:\n{text[:300]}")
    data = json.loads(m.group(0))
    trios = [Trio(t["action"], t["object"], t.get("target")) for t in data["trios"]]
    edges = [(i, i + 1) for i in range(len(trios) - 1)]
    return ITG(trios=trios, edges=edges)


                                                                             
                  
                                                                             
class LARPAgent:
    def __init__(self, model: str = "gemini-2.5-flash", max_repair_rounds: int = 1):
        self.model = model
        self.max_repair_rounds = max_repair_rounds
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            try:
                from google import genai
            except Exception as e:
                raise RuntimeError(

                    "GOOGLE_API_KEY") from e
                                                                             
            self._client = genai.Client()
        return self._client

    def _call(self, command: str, scene: Scene, prior_errors=None) -> str:
        from google.genai import types
        client = self._client_lazy()
        resp = client.models.generate_content(
            model=self.model,
            contents=build_user_prompt(command, scene, prior_errors),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                                                                                         
                response_mime_type="application/json",
            ),
        )
        return resp.text

    def plan(self, command: str, scene: Scene) -> Dict:
        prior_errors = None
        itg = None
        for rnd in range(self.max_repair_rounds + 1):
            raw = self._call(command, scene, prior_errors)
            itg = parse_itg(raw)
            errors = validate_itg(itg, scene)
            if not errors:
                return {"itg": itg, "valid": True, "errors": [], "rounds": rnd}
            prior_errors = errors
        return {"itg": itg, "valid": False, "errors": prior_errors,
                "rounds": self.max_repair_rounds}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--command", default="make a sandwich")
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    scene = Scene(
        objects=["plate_1", "bottom_bread_1", "meat_1", "tomato_1",
                 "top_bread_1", "knife_1", "table_1"],
        on=[("plate_1", "table_1"), ("bottom_bread_1", "table_1"),
            ("meat_1", "table_1"), ("tomato_1", "table_1"),
            ("top_bread_1", "table_1"), ("knife_1", "table_1")],
    )
    agent = LARPAgent(model=args.model)
    result = agent.plan(args.command, scene)
    print(json.dumps({
        "valid": result["valid"], "rounds": result["rounds"],
        "errors": result["errors"], "itg": result["itg"].as_dict(),
    }, indent=2))
