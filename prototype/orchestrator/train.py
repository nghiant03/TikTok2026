"""Training script run inside each sandbox. Contract:
  - reads data via data.load / data.encode (same as baseline)
  - trains a model
  - writes preds_valid.npy and preds_test.npy (float scores, one per evaluation row, original order)
  - prints the same 'valid GAUC .. primary ..' line as baseline for humans

--loss logloss  : identical to official FM baseline (reproduces 0.6015)
--loss bpr      : within-user pairwise loss (hypothesis h1)
Agents modify this file (or baseline.py / data.py) in the sandbox.
"""
import argparse, time, collections
import numpy as np
from data import load, encode
from evaluate import evaluate
from baseline import FM, sigmoid

def apply_grad(m, X, g):
    """Same Adam update as FM.step, but with externally supplied dL/dz per row (g)."""
    z, E, S = m.logits(X)
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()

def build_pairs(users, y):
    """Per-user positive rows and negative rows (only users with both)."""
    pos = collections.defaultdict(list); neg = collections.defaultdict(list)
    for i, (u, l) in enumerate(zip(users, y)):
        (pos if l > 0.5 else neg)[u].append(i)
    P, Nstart, Ncnt, negs = [], [], [], []
    off = 0
    for u in pos:
        if u not in neg:
            continue
        n = neg[u]
        negs.extend(n)
        for p in pos[u]:
            P.append(p); Nstart.append(off); Ncnt.append(len(n))
        off += len(n)
    return np.array(P), np.array(Nstart), np.array(Ncnt), np.array(negs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./KuaiRand-Pure/data")
    ap.add_argument("--loss", default="logloss", choices=["logloss", "bpr"])
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--neg_per_pos", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    splits = load(a.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]; Xva, yva, uva = enc["valid"]; Xte, yte, ute = enc["test"]
    m = FM(dim, k=a.k, lr=a.lr, seed=a.seed)
    rng = np.random.default_rng(a.seed)

    if a.loss == "bpr":
        P, Ns, Nc, NEG = build_pairs(utr, ytr)
        print(f"bpr pairs base: {len(P)} positives with negatives")

    best, best_state, bad = -1, None, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); losses = []
        if a.loss == "logloss":
            idx = rng.permutation(len(ytr))
            for i in range(0, len(idx), a.bs):
                b = idx[i:i + a.bs]
                z = m.logits(Xtr[b])[0]
                p = sigmoid(z)
                apply_grad(m, Xtr[b], ((p - ytr[b]) / len(b)).astype(np.float32))
                losses.append(float(-np.mean(ytr[b] * np.log(p + 1e-9) + (1 - ytr[b]) * np.log(1 - p + 1e-9))))
        else:
            Pr = np.repeat(P, a.neg_per_pos); Nsr = np.repeat(Ns, a.neg_per_pos); Ncr = np.repeat(Nc, a.neg_per_pos)
            order = rng.permutation(len(Pr))
            for i in range(0, len(order), a.bs):
                b = order[i:i + a.bs]
                pi = Pr[b]; ni = NEG[Nsr[b] + rng.integers(Ncr[b])]
                zp = m.logits(Xtr[pi])[0]; zn = m.logits(Xtr[ni])[0]
                d = zp - zn; s = sigmoid(d)
                gp = (s - 1) / len(b); gn = (1 - s) / len(b)
                apply_grad(m, np.concatenate([Xtr[pi], Xtr[ni]]),
                           np.concatenate([gp, gn]).astype(np.float32))
                losses.append(float(-np.mean(np.log(s + 1e-9))))
        va = evaluate(uva, yva, m.predict(Xva))
        print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
              f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s", flush=True)
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep}"); break
    m.V, m.W, m.b = best_state
    pv, pt = m.predict(Xva), m.predict(Xte)
    np.save("preds_valid.npy", pv.astype(np.float32)); np.save("preds_test.npy", pt.astype(np.float32))
    r = evaluate(uva, yva, pv)
    print(f"\nFINAL valid GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")

if __name__ == "__main__":
    main()
