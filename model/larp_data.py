

import os, json, glob
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data, Dataset


                                                                             
            
                                                                             

class Vocab:
    def __init__(self, dataset_dir: str):
        vd = os.path.join(dataset_dir, "vocab")
        objs = json.load(open(os.path.join(vd, "objects.json")))
        acts = json.load(open(os.path.join(vd, "actions.json")))
        rels = json.load(open(os.path.join(vd, "relations.json")))
        tax  = json.load(open(os.path.join(vd, "taxonomy.json")))

                                                                                  
        self.obj_classes = ["<unk>"] + sorted(objs["object_affordances"].keys())
        self.obj2idx = {o: i for i, o in enumerate(self.obj_classes)}

        self.affordances = objs["affordances"]
        self.aff2idx = {a: i for i, a in enumerate(self.affordances)}
        self.obj_affordances = objs["object_affordances"]

        self.state_keys = acts["state_keys"]                                     
        self.actions = ["<unk>"] + acts["actions"]
        self.act2idx = {a: i for i, a in enumerate(self.actions)}

        self.relations = ["<unk>"] + rels["relations"]
        self.rel2idx = {r: i for i, r in enumerate(self.relations)}

        self.categories = ["none"] + tax["categories"]                         
        self.cat2idx = {c: i for i, c in enumerate(self.categories)}

                                                                                                          
    @property
    def node_dim(self):
        return len(self.obj_classes) + len(self.affordances) + 4 + 2

    @property
    def edge_dim(self):
        return len(self.relations)

    @property
    def trio_dim(self):                                                        
        return len(self.actions)


def _obj_class_idx(v: Vocab, cls: str) -> int:
    return v.obj2idx.get(cls, v.obj2idx["<unk>"])


                                                                             
                      
                                                                             

def step_to_data(step: Dict, vocab: Vocab, use_observed: bool = True) -> Optional[Data]:
                                                                         
                                             
    g = step.get("state_before_observed") if use_observed else None
    if g is None:
        g = step["state_before"]
    nodes = g["nodes"]
    if not nodes:
        return None
    trio = step["trio"]
    q_obj = trio["object"]
    q_tgt = trio.get("target")

    id2idx = {n["id"]: i for i, n in enumerate(nodes)}

                   
    feats = []
    for n in nodes:
        f = torch.zeros(vocab.node_dim)
        off = 0
                       
        f[off + _obj_class_idx(vocab, n["type"])] = 1.0
        off += len(vocab.obj_classes)
                              
        for a in vocab.obj_affordances.get(n["type"], []):
            if a in vocab.aff2idx:
                f[off + vocab.aff2idx[a]] = 1.0
        off += len(vocab.affordances)
                                                                    
        st = n["state"]
        f[off + 0] = 1.0 if st.get("held") else 0.0
        f[off + 1] = 1.0 if st.get("open") else 0.0
        f[off + 2] = 1.0 if st.get("sliced") else 0.0
        f[off + 3] = 1.0 if st.get("clear") else 0.0
        off += 4
                    
        f[off + 0] = 1.0 if n["id"] == q_obj else 0.0
        f[off + 1] = 1.0 if (q_tgt is not None and n["id"] == q_tgt) else 0.0
        feats.append(f)
    x = torch.stack(feats, 0)

                                
    src, dst, eattr = [], [], []
    for e in g["edges"]:
        if e["subject"] in id2idx and e["object"] in id2idx:
            src.append(id2idx[e["subject"]]); dst.append(id2idx[e["object"]])
            ef = torch.zeros(vocab.edge_dim)
            ef[vocab.rel2idx.get(e["relation"], vocab.rel2idx["<unk>"])] = 1.0
            eattr.append(ef)
    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.stack(eattr, 0)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, vocab.edge_dim)

                        
    trio_vec = torch.zeros(vocab.trio_dim)
    trio_vec[vocab.act2idx.get(trio["action"], vocab.act2idx["<unk>"])] = 1.0

             
    lab = step["labels"]
    y_exec = torch.tensor([1 if lab["violation"] else 0], dtype=torch.long)
    y_cat = torch.tensor([vocab.cat2idx.get(lab["failure_category"] or "none", 0)],
                         dtype=torch.long)
                                                             
    blk = lab.get("blocking_object")
    blk_idx = id2idx.get(blk, -1) if blk is not None else -1
    y_blk = torch.tensor([blk_idx], dtype=torch.long)
                       
    br = lab.get("blocking_relation")
    y_rel = torch.tensor([vocab.rel2idx.get(br, 0) if br else 0], dtype=torch.long)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    d.trio_vec = trio_vec.unsqueeze(0)
    d.y_exec = y_exec
    d.y_cat = y_cat
    d.y_blk = y_blk
    d.y_rel = y_rel
    d.num_nodes_real = torch.tensor([x.size(0)])
    return d


                                                                             
                      
                                                                             

class LARPSteps(Dataset):
    def __init__(self, dataset_dir: str, split: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        ids = [l.strip() for l in
               open(os.path.join(dataset_dir, "splits", f"{split}.txt"))
               if l.strip()]
        idset = set(ids)
        self.samples: List[Data] = []
        for path in glob.glob(os.path.join(dataset_dir, "episodes", "*", "*.json")):
            ep = json.load(open(path))
            if ep["episode_id"] not in idset:
                continue
            for step in ep["steps"]:
                d = step_to_data(step, vocab)
                if d is not None:
                    self.samples.append(d)

    def len(self): return len(self.samples)
    def get(self, i): return self.samples[i]
