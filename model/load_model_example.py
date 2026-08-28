
import sys, torch
sys.path.append(".")                                                                      
from larp_data import Vocab, step_to_data
from larp_model import GATVerifier

DATASET = "../larp_dataset"                                             
CKPT    = "runs/gat_best.pt"

vocab = Vocab(DATASET)
model = GATVerifier(vocab.node_dim, vocab.edge_dim, vocab.trio_dim,
                    len(vocab.categories), len(vocab.relations))
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
model.eval()
print("model loaded:", sum(p.numel() for p in model.parameters()), "params")

                                                                   
                                                                             
