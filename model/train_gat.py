

from html import parser
import os, json, argparse, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from larp_data import Vocab, LARPSteps, step_to_data
from larp_model import GATVerifier, symbolic_predict


def multitask_loss(out, batch, w_cat=0.5, w_blk=0.5, w_rel=0.3):
    y_exec = batch.y_exec.view(-1)
    y_cat = batch.y_cat.view(-1)
    y_rel = batch.y_rel.view(-1)
    loss = F.cross_entropy(out["exec"], y_exec)
    loss = loss + w_cat * F.cross_entropy(out["cat"], y_cat)
    loss = loss + w_rel * F.cross_entropy(out["rel"], y_rel)
                                                                      
    y_blk = batch.y_blk.view(-1)                                                
    has_blk = (y_blk >= 0)
    if has_blk.any():
                                                             
                                
        counts = batch.num_nodes_real.view(-1)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=counts.device),
                             torch.cumsum(counts, 0)[:-1]])
                                           
        blk_loss = 0.0; n = 0
        for gi in range(counts.size(0)):
            if y_blk[gi] < 0:
                continue
            start = int(offsets[gi]); end = start + int(counts[gi])
            scores = out["blk_score"][start:end].unsqueeze(0)           
            target = y_blk[gi].view(1)
            blk_loss = blk_loss + F.cross_entropy(scores, target)
            n += 1
        if n > 0:
            loss = loss + w_blk * (blk_loss / n)
    return loss


@torch.no_grad()
def eval_gat(model, loader, device):
    model.eval()
    P, Y = [], []
    blk_correct = blk_total = 0
    cat_correct = cat_total = 0
    counts_cum = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        pred = out["exec"].argmax(1).cpu()
        P += pred.tolist(); Y += batch.y_exec.view(-1).cpu().tolist()
                                                           
        catp = out["cat"].argmax(1).cpu(); caty = batch.y_cat.view(-1).cpu()
        for a, b in zip(catp.tolist(), caty.tolist()):
            if b != 0:
                cat_total += 1; cat_correct += int(a == b)
                     
        counts = batch.num_nodes_real.view(-1)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long),
                             torch.cumsum(counts.cpu(), 0)[:-1]])
        yblk = batch.y_blk.view(-1).cpu()
        bs = out["blk_score"].cpu()
        for gi in range(counts.size(0)):
            if yblk[gi] < 0: continue
            s = int(offsets[gi]); e = s + int(counts[gi])
            pidx = int(bs[s:e].argmax())
            blk_total += 1; blk_correct += int(pidx == int(yblk[gi]))
    p, r, f, _ = precision_recall_fscore_support(Y, P, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"P": p, "R": r, "F1": f,
            "blk_acc": blk_correct / max(blk_total, 1),
            "cat_acc": cat_correct / max(cat_total, 1),
            "n": len(Y)}


def eval_symbolic(dataset_dir, split, vocab, use_observed=True):
    
    import glob
    ids = set(l.strip() for l in
              open(os.path.join(dataset_dir, "splits", f"{split}.txt")) if l.strip())
    P, Y = [], []
    for path in glob.glob(os.path.join(dataset_dir, "episodes", "*", "*.json")):
        ep = json.load(open(path))
        if ep["episode_id"] not in ids: continue
        for step in ep["steps"]:
            pred = symbolic_predict(step, use_observed=use_observed)
            P.append(1 if pred["violation"] else 0)
            Y.append(1 if step["labels"]["violation"] else 0)
    p, r, f, _ = precision_recall_fscore_support(Y, P, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"P": p, "R": r, "F1": f, "n": len(Y)}


def main():
    ap = argparse.ArgumentParser()
    parser.add_argument("--use_gcn", action="store_true")
    parser.add_argument("--ablate_func_state", action="store_true")
    parser.add_argument("--ablate_query", action="store_true")
    parser.add_argument("--single_head", action="store_true")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--out", default="./runs")
    
    args = ap.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] device: {device}")
    os.makedirs(args.out, exist_ok=True)

    vocab = Vocab(args.dataset)
    train_ds = LARPSteps(args.dataset, "train", vocab)
    val_ds = LARPSteps(args.dataset, "val", vocab)
    print(f"[+] train steps: {len(train_ds)}  val steps: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = GATVerifier(
        node_dim=vocab.node_dim, edge_dim=vocab.edge_dim, trio_dim=vocab.trio_dim,
        n_categories=len(vocab.categories), n_relations=len(vocab.relations),
        hidden=args.hidden, heads=args.heads, layers=args.layers,use_gcn=args.use_gcn,
                    ablate_func_state=args.ablate_func_state,
                    ablate_query=args.ablate_query,
                    single_head=args.single_head,dropout=args.dropout).to(device)
    print(f"[+] params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=5)
    best_f1, best_state, no_imp = -1, None, 0
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            loss = multitask_loss(out, batch)
            loss.backward(); opt.step(); tot += float(loss)
        vm = eval_gat(model, val_loader, device)
        sched.step(vm["F1"])
        if ep % 5 == 0 or ep == 1:
            print(f"  ep {ep:3d}  loss={tot/len(train_loader):.3f}  "
                  f"val F1={vm['F1']:.3f} blk_acc={vm['blk_acc']:.3f} cat_acc={vm['cat_acc']:.3f}")
        if vm["F1"] > best_f1:
            best_f1, best_state, no_imp = vm["F1"], copy.deepcopy(model.state_dict()), 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print(f"  early stop @ {ep}"); break
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(args.out, "gat_best.pt"))

                                              
    print("\n" + "=" * 64)
    print(f"{'split':<22}{'method':<10}{'P':>7}{'R':>7}{'F1':>7}{'blk':>7}{'cat':>7}")
    print("=" * 64)
    results = {}
    for split in ["test_seen", "test_unseen_objects", "test_unseen_configs"]:
        ds = LARPSteps(args.dataset, split, vocab)
        loader = DataLoader(ds, batch_size=args.batch_size)
        gm = eval_gat(model, loader, device)
        sm_obs = eval_symbolic(args.dataset, split, vocab, use_observed=True)
        sm_oracle = eval_symbolic(args.dataset, split, vocab, use_observed=False)
        results[split] = {"gat": gm, "symbolic_observed": sm_obs,
                          "symbolic_oracle": sm_oracle}
        print(f"{split:<22}{'GAT(obs)':<14}{gm['P']:>7.3f}{gm['R']:>7.3f}{gm['F1']:>7.3f}"
              f"{gm['blk_acc']:>7.3f}{gm['cat_acc']:>7.3f}")
        print(f"{'':<22}{'symbolic(obs)':<14}{sm_obs['P']:>7.3f}{sm_obs['R']:>7.3f}{sm_obs['F1']:>7.3f}")
        print(f"{'':<22}{'symbolic(oracle)':<14}{sm_oracle['P']:>7.3f}{sm_oracle['R']:>7.3f}{sm_oracle['F1']:>7.3f}")
    print("=" * 64)
                                                                       
    for split in ["test_seen", "test_unseen_objects", "test_unseen_configs"]:
        gap = results[split]["gat"]["F1"] - results[split]["symbolic_observed"]["F1"]
        print(f"[realistic gap] {split}: GAT {results[split]['gat']['F1']:.3f} "
              f"vs symbolic(obs) {results[split]['symbolic_observed']['F1']:.3f}  ({gap:+.3f})")
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print(f"\n[+] saved model + results to {args.out}")


if __name__ == "__main__":
    main()
