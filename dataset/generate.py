
import os
import json
import argparse
import random
from typing import List, Dict, Tuple, Optional

from domain import (Trio, FAILURE_CATEGORIES, ACTIONS, RELATIONS,
                    STATE_KEYS, AFFORDANCES, OBJECT_AFFORDANCES)
from worldstate import WorldState
from labeler import roll_forward, CORRUPTORS


def _table_scene(objects: List[Tuple[str, str]]) -> WorldState:
    s = WorldState()
    s.add_object("table_1", "table")
    for inst, cls in objects:
        s.add_object(inst, cls)
        s.add_on(inst, "table_1")
    return s


def task_make_sandwich() -> Tuple[WorldState, List[Trio], List[str]]:
    s = _table_scene([
        ("plate_1", "plate"), ("bottom_bread_1", "bottom_bread"),
        ("meat_1", "meat"), ("tomato_1", "tomato"), ("top_bread_1", "top_bread"),
        ("knife_1", "knife")])
    plan = [
        Trio("pick", "bottom_bread_1"), Trio("put", "bottom_bread_1", "plate_1"),
        Trio("pick", "meat_1"),         Trio("put", "meat_1", "bottom_bread_1"),
        Trio("pick", "tomato_1"),       Trio("put", "tomato_1", "meat_1"),
        Trio("pick", "top_bread_1"),    Trio("put", "top_bread_1", "tomato_1"),
    ]
    inventory = ["plate", "bottom_bread", "meat", "tomato", "top_bread", "knife"]
    return s, plan, inventory


def task_make_salad() -> Tuple[WorldState, List[Trio], List[str]]:
    s = _table_scene([
        ("bowl_1", "bowl"), ("lettuce_1", "lettuce"),
        ("tomato_1", "tomato"), ("knife_1", "knife")])
    plan = [
        Trio("pick", "knife_1"),
        Trio("slice", "tomato_1"),
        Trio("put", "knife_1", "table_1"),
        Trio("pick", "lettuce_1"), Trio("put", "lettuce_1", "bowl_1"),
        Trio("pick", "tomato_1"),  Trio("put", "tomato_1", "bowl_1"),
    ]
    inventory = ["bowl", "lettuce", "tomato", "knife"]
    return s, plan, inventory


def task_clear_and_plate() -> Tuple[WorldState, List[Trio], List[str]]:
    s = _table_scene([("plate_1", "plate"), ("apple_1", "apple")])
    plan = [Trio("pick", "apple_1"), Trio("put", "apple_1", "plate_1")]
    inventory = ["plate", "apple"]
    return s, plan, inventory


def task_fetch_from_fridge() -> Tuple[WorldState, List[Trio], List[str]]:
    s = _table_scene([("plate_1", "plate"), ("cheese_1", "cheese")])
    s.add_object("fridge_1", "fridge", is_open=True)
    s.add_object("milk_1", "milk")
    s.add_on("milk_1", "table_1")
    plan = [
        Trio("pick", "milk_1"), Trio("put", "milk_1", "table_1"),
        Trio("pick", "cheese_1"), Trio("put", "cheese_1", "plate_1"),
    ]
    inventory = ["plate", "cheese", "fridge", "milk"]
    return s, plan, inventory


TASK_TEMPLATES = {
    "make_sandwich": task_make_sandwich,
    "make_salad": task_make_salad,
    "clear_and_plate": task_clear_and_plate,
    "fetch_from_fridge": task_fetch_from_fridge,
}


def steplabels_to_json(labels) -> List[Dict]:
    out = []
    for l in labels:
        out.append({
            "step_index": l.step_index,
            "trio": l.trio.as_dict(),
            "state_before": l.state_before,
            "state_after": l.state_after,
            "labels": {
                "violation": l.violation,
                "failure_category": l.failure_category,
                "blocking_object": l.blocking_object,
                "blocking_relation": l.blocking_relation,
                "gold_repair": l.gold_repair,
            },
        })
    return out


def make_complete_episode(task_id: str, scene_id: str,
                          initial: WorldState, plan: List[Trio],
                          inventory: List[str], ep_idx: int) -> Dict:
    labels = roll_forward(initial, plan)
    return {
        "episode_id": f"{task_id}_{scene_id}_complete_{ep_idx:04d}",
        "task_id": task_id, "scene_id": scene_id,
        "label": "complete", "corruption_type": None,
        "object_inventory": inventory, "config_id": "canonical",
        "reference_plan": [t.as_dict() for t in plan],
        "plan": [t.as_dict() for t in plan],
        "steps": steplabels_to_json(labels),
    }


def make_corrupted_episode(task_id: str, scene_id: str,
                           initial: WorldState, plan: List[Trio],
                           inventory: List[str], corruption_type: str,
                           rng: random.Random, ep_idx: int) -> Optional[Dict]:
    out = CORRUPTORS[corruption_type](initial, plan, rng)
    if out is None:
        return None
    cinit, cplan, meta = out
    labels = roll_forward(cinit, cplan)
    if not any(l.violation for l in labels):
        return None
    config_id = meta.get("on_target") or meta.get("container") or\
                meta.get("corruption_type")
    return {
        "episode_id": f"{task_id}_{scene_id}_{corruption_type}_{ep_idx:04d}",
        "task_id": task_id, "scene_id": scene_id,
        "label": "corrupted", "corruption_type": corruption_type,
        "object_inventory": inventory, "config_id": str(config_id),
        "corruption_meta": meta,
        "reference_plan": [t.as_dict() for t in plan],
        "plan": [t.as_dict() for t in cplan],
        "steps": steplabels_to_json(labels),
    }


def write_vocab(out_dir: str):
    vd = os.path.join(out_dir, "vocab")
    os.makedirs(vd, exist_ok=True)
    json.dump({"object_affordances": OBJECT_AFFORDANCES, "affordances": AFFORDANCES},
              open(os.path.join(vd, "objects.json"), "w"), indent=2)
    json.dump({"actions": ACTIONS, "state_keys": STATE_KEYS},
              open(os.path.join(vd, "actions.json"), "w"), indent=2)
    json.dump({"relations": RELATIONS},
              open(os.path.join(vd, "relations.json"), "w"), indent=2)
    json.dump({"categories": FAILURE_CATEGORIES},
              open(os.path.join(vd, "taxonomy.json"), "w"), indent=2)


def build_splits(manifest: List[Dict], out_dir: str, rng: random.Random,
                 heldout_objects=None, heldout_configs=None):
    heldout_objects = set(heldout_objects or {"cheese"})
    heldout_configs = set(heldout_configs or {"containment"})

    train, val, seen, unseen_obj, unseen_cfg = [], [], [], [], []
    for m in manifest:
        eid = m["episode_id"]
        inv = set(m["object_inventory"])
        if m.get("corruption_type") in heldout_configs:
            unseen_cfg.append(eid); continue
        if inv & heldout_objects:
            unseen_obj.append(eid); continue
        r = rng.random()
        if r < 0.7: train.append(eid)
        elif r < 0.85: val.append(eid)
        else: seen.append(eid)

    sd = os.path.join(out_dir, "splits"); os.makedirs(sd, exist_ok=True)
    for name, ids in [("train", train), ("val", val), ("test_seen", seen),
                      ("test_unseen_objects", unseen_obj),
                      ("test_unseen_configs", unseen_cfg)]:
        with open(os.path.join(sd, f"{name}.txt"), "w") as f:
            f.write("\n".join(ids))
    return {k: len(v) for k, v in
            [("train", train), ("val", val), ("test_seen", seen),
             ("test_unseen_objects", unseen_obj), ("test_unseen_configs", unseen_cfg)]}


def _ground_in_thor(thor, scene_id, sym_initial, sym_plan,
                    worldstate_from_metadata, helpers, rng, rgb_out_dir):
    from domain import Trio

    sym_ids = set()
    for t in sym_plan:
        sym_ids.add(t.object)
        if t.target and t.target != "clear_location":
            sym_ids.add(t.target)
    sym_ids |= set(sym_initial.cls.keys())

    id_to_class = {sid: sym_initial.cls.get(sid, sid.split("_")[0]) for sid in sym_ids}
    required_types = [f"{cls}#{i}" for i, (sid, cls) in enumerate(id_to_class.items())]
    req_key_to_symid = {f"{cls}#{i}": sid
                        for i, (sid, cls) in enumerate(id_to_class.items())}

    thor.reset(scene_id)
    objs = thor.objects()
    chosen = helpers["resolve"](objs, required_types, rng)
    if chosen is None:
        return None

    symid_to_thor = {req_key_to_symid[k]: v for k, v in chosen.items()}
    thor_to_symid = {v: k for k, v in symid_to_thor.items()}

    full_state, id_map = worldstate_from_metadata(objs)
    grounded = sym_initial.clone()
    for o in objs:
        tid = o["objectId"]
        if tid in thor_to_symid:
            sid = thor_to_symid[tid]
            if "position" in o:
                grounded.pos[sid] = [o["position"]["x"], o["position"]["y"], o["position"]["z"]]
            grounded.set_open(sid, bool(o.get("isOpen", False)))
            if o.get("isSliced", False):
                grounded.set_sliced(sid, True)

    grounded_plan = list(sym_plan)

    id_map_rev = symid_to_thor
    for step_i, t in enumerate(grounded_plan):
        action = helpers["trio_to_action"](t, id_map_rev)
        if action is None:
            continue
        thor.step(**action)
        if rgb_out_dir:
            d = os.path.join(rgb_out_dir, "images", f"{scene_id}")
            os.makedirs(d, exist_ok=True)
            helpers["capture_rgb"](thor, os.path.join(d, f"step_{step_i:03d}.png"))

    return grounded, grounded_plan


def generate(out_dir: str, mode: str, n_per_task: int, seed: int,
             save_rgb: bool = False, noise: float = 0.0):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    write_vocab(out_dir)

    thor = None
    worldstate_from_metadata = None
    thor_helpers = None
    if mode == "thor":
        from thor_adapter import (ThorSession, worldstate_from_metadata,
                                  resolve_required_types, trio_to_thor_action,
                                  capture_rgb_png)
        thor = ThorSession()
        thor_helpers = {
            "resolve": resolve_required_types,
            "trio_to_action": trio_to_thor_action,
            "capture_rgb": capture_rgb_png,
        }

    manifest: List[Dict] = []
    counts = {"complete": 0, "corrupted": 0}

    for task_id, builder in TASK_TEMPLATES.items():
        for rep in range(n_per_task):
            initial, plan, inventory = builder()

            if mode == "symbolic":
                scene_id = "symbolic"
                grounded_initial, grounded_plan = initial, plan
            else:
                scene_id = f"FloorPlan{(rep % 30) + 1}"
                grounded = _ground_in_thor(thor, scene_id, initial, plan,
                                           worldstate_from_metadata,
                                           thor_helpers, rng,
                                           out_dir if save_rgb else None)
                if grounded is None:
                    continue
                grounded_initial, grounded_plan = grounded

            ep = make_complete_episode(task_id, scene_id, grounded_initial,
                                       grounded_plan, inventory, rep)
            if noise > 0:
                from noise import noisy_episode
                ep = noisy_episode(ep, rng, noise)
            _write_episode(out_dir, ep); manifest.append(_manifest_row(ep)); counts["complete"] += 1

            for cat in FAILURE_CATEGORIES:
                cep = make_corrupted_episode(task_id, scene_id, grounded_initial,
                                             grounded_plan, inventory, cat, rng, rep)
                if cep is not None:
                    if noise > 0:
                        from noise import noisy_episode
                        cep = noisy_episode(cep, rng, noise)
                    _write_episode(out_dir, cep); manifest.append(_manifest_row(cep))
                    counts["corrupted"] += 1

    if thor is not None:
        thor.stop()

    with open(os.path.join(out_dir, "manifest.jsonl"), "w") as f:
        for row in manifest:
            f.write(json.dumps(row) + "\n")

    split_counts = build_splits(manifest, out_dir, rng)

    print(f"[+] wrote {counts['complete']} complete, {counts['corrupted']} corrupted episodes")
    print(f"[+] splits: {split_counts}")
    print(f"[+] output: {out_dir}")


def _write_episode(out_dir: str, ep: Dict):
    d = os.path.join(out_dir, "episodes", ep["task_id"]); os.makedirs(d, exist_ok=True)
    json.dump(ep, open(os.path.join(d, ep["episode_id"] + ".json"), "w"), indent=2)


def _manifest_row(ep: Dict) -> Dict:
    return {k: ep[k] for k in
            ["episode_id", "task_id", "scene_id", "label",
             "corruption_type", "object_inventory", "config_id"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./larp_dataset")
    ap.add_argument("--mode", choices=["symbolic", "thor"], default="symbolic")
    ap.add_argument("--n_per_task", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_rgb", action="store_true",
                    help="(thor mode) save one RGB frame per step")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="perception-noise scale in [0,1]; adds observed graphs")
    args = ap.parse_args()
    generate(args.out_dir, args.mode, args.n_per_task, args.seed,
             args.save_rgb, args.noise)
