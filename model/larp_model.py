

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, global_max_pool


class GATVerifier(nn.Module):
    def __init__(self, node_dim, edge_dim, trio_dim,
                 n_categories, n_relations,
                 hidden=128, heads=4, layers=3, dropout=0.3,
                                           
                 use_gcn=False,
                 ablate_func_state=False,
                 ablate_query=False,
                 single_head=False,
                                                                               
                                                                         
                                                                        
                                                                
                 func_state_slice=slice(-4, None)):
        super().__init__()
        self.dropout = dropout
        self.use_gcn = use_gcn
        self.ablate_func_state = ablate_func_state
        self.ablate_query = ablate_query
        self.single_head = single_head
        self.func_state_slice = func_state_slice
        self.n_categories = n_categories
        self.n_relations = n_relations

                                           
        self.convs = nn.ModuleList()
        in_dim = node_dim
        for i in range(layers):
            out = hidden
            if use_gcn:
                                                                      
                                                                            
                self.convs.append(GCNConv(in_dim, out))
                in_dim = out
            else:
                self.convs.append(
                    GATv2Conv(in_dim, out, heads=heads, edge_dim=edge_dim,
                              dropout=dropout, concat=True))
                in_dim = out * heads
        self.node_out = in_dim                                       

                                        
        self.trio_enc = nn.Sequential(
            nn.Linear(trio_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

                                                                               
        ctx = self.node_out * 2 + hidden

                                
        if single_head:
                                                                          
                                                                           
                                                                          
                                                                        
            self.unified_head = nn.Sequential(
                nn.Linear(ctx, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2 + n_categories + n_relations))
        else:
            self.exec_head = nn.Sequential(
                nn.Linear(ctx, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, 2))
            self.cat_head = nn.Sequential(
                nn.Linear(ctx, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_categories))
            self.rel_head = nn.Sequential(
                nn.Linear(ctx, hidden), nn.ReLU(),
                nn.Linear(hidden, n_relations))

                                                                                  
                                                  
        self.blk_head = nn.Sequential(
            nn.Linear(self.node_out + ctx, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, batch):
        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr

                                                                 
        if self.ablate_func_state:
            x = x.clone()
            x[:, self.func_state_slice] = 0

                                      
        for conv in self.convs:
            if self.use_gcn:
                                                 
                x = F.elu(conv(x, ei))
            else:
                x = F.elu(conv(x, ei, ea))
            x = F.dropout(x, p=self.dropout, training=self.training)

        gmean = global_mean_pool(x, batch.batch)
        gmax = global_max_pool(x, batch.batch)

                                          
        trio = self.trio_enc(batch.trio_vec.squeeze(1))
                                                       
        if self.ablate_query:
            trio = torch.zeros_like(trio)

        ctx = torch.cat([gmean, gmax, trio], dim=1)                  

                                
        if self.single_head:
            out = self.unified_head(ctx)
            exec_logits = out[:, :2]
            cat_logits = out[:, 2:2 + self.n_categories]
            rel_logits = out[:, 2 + self.n_categories:]
        else:
            exec_logits = self.exec_head(ctx)
            cat_logits = self.cat_head(ctx)
            rel_logits = self.rel_head(ctx)

                                                               
        ctx_per_node = ctx[batch.batch]                              
        blk_in = torch.cat([x, ctx_per_node], dim=1)                          
        blk_score = self.blk_head(blk_in).squeeze(-1)           

        return {
            "exec": exec_logits, "cat": cat_logits,
            "rel": rel_logits, "blk_score": blk_score,
        }


                                                                             
                                                                       
                                                                        
                                                                         
                                                         
                                                                     
                                                                             

def symbolic_predict(step: dict, use_observed: bool = True) -> dict:
    
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from worldstate import WorldState
    from domain import PRECONDITIONS, Trio

    g = None
    if use_observed:
        g = step.get("state_before_observed")
    if g is None:
        g = step["state_before"]
    s = WorldState()
    for n in g["nodes"]:
        s.add_object(n["id"], n["type"],
                     is_open=bool(n["state"].get("open")))
        if n["state"].get("held"): s.set_held(n["id"], True)
        if n["state"].get("sliced"): s.set_sliced(n["id"], True)
    for e in g["edges"]:
        if e["relation"] == "on": s.add_on(e["subject"], e["object"])
        elif e["relation"] == "in": s.add_in(e["subject"], e["object"])

    t = Trio.from_dict(step["trio"])
    pre = PRECONDITIONS.get(t.action)
    v = pre(s, t) if pre else None
    if v is None:
        return {"violation": False, "category": "none",
                "blocker": None, "rel": None}
    return {"violation": True, "category": v.category,
            "blocker": v.blocking_object, "rel": v.blocking_relation}