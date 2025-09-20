import argparse, json, os
from collections import deque
from functools import lru_cache

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from core import PolicyNet, check_winner, encode_state

def legal_moves(board): 
    return [i for i,v in enumerate(board) if v == 0]

def _to_move_from_board(board):
    x = sum(1 for v in board if v==1)
    o = sum(1 for v in board if v==-1)
    return 1 if x == o else -1

@lru_cache(maxsize=None)
def _minimax_score(state_tuple, player):
    """
    Return best outcome for 'player' on 'state_tuple':
      +1 win, 0 draw, -1 loss
    """
    b = list(state_tuple)
    term = check_winner(b)
    if term is not None:
        return 0 if term == 0 else (1 if term == player else -1)

    best = -2
    for a in legal_moves(b):
        b[a] = player
        s = -_minimax_score(tuple(b), -player)
        b[a] = 0
        if s > best:
            best = s
            if best == 1:
                break
    return best

def optimal_moves_minimax(board, player):
    """(optimal_actions, best_score) for 'player' on 'board'."""
    best, acts = -2, []
    for a in legal_moves(board):
        board[a] = player
        s = -_minimax_score(tuple(board), -player)
        board[a] = 0
        if s > best:
            best, acts = s, [a]
        elif s == best:
            acts.append(a)
    return acts, best

def generate_all_positions():
    seen = set()
    q = deque([[0]*9])
    out = []
    while q:
        b = q.popleft()
        key = tuple(b)
        if key in seen: 
            continue
        seen.add(key)
        out.append(b[:])

        if check_winner(b) is not None:
            continue
        to = _to_move_from_board(b)
        for a in legal_moves(b):
            b[a] = to
            q.append(b[:])
            b[a] = 0
    return out

# ---------- Build supervised dataset ----------
def build_dataset():
    boards = generate_all_positions()
    # X: [18], Y: distribution over optimal legal moves
    X_list, Y_list = [], []

    for b in boards:
        if check_winner(b) is not None:
            continue
        to = _to_move_from_board(b)
        bb = b[:] if to == 1 else [-v for v in b]  # flip so +1 is side-to-move

        opts, _ = optimal_moves_minimax(bb, 1)
        if not opts:
            continue

        x = encode_state(bb)                      # np.ndarray shape (18,)
        y = np.zeros(9, dtype=np.float32)         # np.ndarray shape (9,)
        w = 1.0 / len(opts)
        for a in opts:
            y[a] = w

        X_list.append(x)
        Y_list.append(y)

    X = torch.from_numpy(np.asarray(X_list, dtype=np.float32))  # shape [N, 18]
    Y = torch.from_numpy(np.asarray(Y_list, dtype=np.float32))  # shape [N, 9]
    return X, Y


def train(epochs=12, batch_size=256, lr=1e-3, out='ttt_model.pt',
                     log_every=1, plot_loss='loss_curve.png'):
    X, Y = build_dataset()
    print(json.dumps({"dataset_size": int(X.size(0))}))

    model = PolicyNet()
    if os.path.exists(out):
        try:
            model.load_state_dict(torch.load(out, map_location='cpu'))
            print(json.dumps({"resumed_from": out}))
        except Exception as e:
            print(json.dumps({"resume_failed": str(e)}))

    opt = optim.Adam(model.parameters(), lr=lr)
    losses = []

    for ep in range(1, epochs+1):
        idx = torch.randperm(X.size(0))
        total = 0.0
        for i in range(0, X.size(0), batch_size):
            xb = X[idx[i:i+batch_size]]
            yb = Y[idx[i:i+batch_size]]
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            loss = -(yb * torch.log(probs + 1e-12)).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        avg = total / X.size(0)
        losses.append(avg)
        if ep % log_every == 0 or ep == 1:
            print(json.dumps({"epoch": ep, "loss": avg}))

    torch.save(model.state_dict(), out)
    print(json.dumps({"saved": out, "epochs": epochs}))

    # Plot loss
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(plot_loss)
    plt.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--out', type=str, default='ttt_model.pt')
    p.add_argument('--log_every', type=int, default=1)
    args = p.parse_args()

    train(epochs=args.epochs,
          batch_size=args.batch_size,
          lr=args.lr,
          out=args.out,
          log_every=args.log_every)
