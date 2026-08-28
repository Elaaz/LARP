

import copy
import random
from typing import Dict, List


def add_perception_noise(graph: Dict, rng: random.Random,
                         p_edge_drop: float = 0.20,
                         p_edge_add: float = 0.05,
                         p_state_flip: float = 0.15) -> Dict:
    
    g = copy.deepcopy(graph)
    node_ids = [n["id"] for n in g["nodes"]]

                                    
    kept = []
    for e in g["edges"]:
        if rng.random() < p_edge_drop:
            continue                    
        kept.append(e)
                                       
    if len(node_ids) >= 2:
        n_add = sum(1 for _ in g["edges"] if rng.random() < p_edge_add)
        for _ in range(n_add):
            a, b = rng.sample(node_ids, 2)
            rel = rng.choice(["on", "in"])
            kept.append({"subject": a, "relation": rel, "object": b})
    g["edges"] = kept

                                            
    for n in g["nodes"]:
        st = n["state"]
        for k in ["held", "open", "sliced", "clear"]:
            if st.get(k) is None:
                continue
            if rng.random() < p_state_flip:
                st[k] = (not st[k])
    return g


def noisy_episode(ep: Dict, rng: random.Random, scale: float) -> Dict:
    
    pe, pa, ps = 0.20 * scale, 0.05 * scale, 0.15 * scale
    for step in ep["steps"]:
        step["state_before_observed"] = add_perception_noise(
            step["state_before"], rng, pe, pa, ps)
                                                                               
        if "state_after" in step:
            step["state_after_observed"] = add_perception_noise(
                step["state_after"], rng, pe, pa, ps)
    ep["perception_noise_scale"] = scale
    return ep
