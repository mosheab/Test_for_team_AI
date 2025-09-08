import random
import numpy as np
import torch
import torch.nn as nn

WIN_LINES = [
    (0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6),
]

def check_winner(board):
    for a,b,c in WIN_LINES:
        s = board[a]+board[b]+board[c]
        if s==3: return 1
        if s==-3: return -1
    if 0 not in board: return 0
    return None

def legal_moves(board):
    return [i for i,v in enumerate(board) if v==0]

def encode_state(board):
    b = np.array(board, dtype=np.int8)
    x = (b==1).astype(np.float32)
    o = (b==-1).astype(np.float32)
    return np.concatenate([x,o], axis=0)

def heuristic_move(board, player):
    opp = -player
    # win
    for m in legal_moves(board):
        b2 = board[:]
        b2[m] = player
        if check_winner(b2) == player: return m
    # block
    for m in legal_moves(board):
        b2 = board[:]
        b2[m] = opp
        if check_winner(b2) == opp: return m
    # center
    if board[4]==0: return 4
    # corners
    corners = [i for i in [0,2,6,8] if board[i]==0]
    if corners: 
        return random.choice(corners)
    # random
    moves = legal_moves(board)
    return random.choice(moves) if moves else None

class PolicyNet(nn.Module):
    def __init__(self, in_dim=18, hidden=64, out_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): 
        return self.net(x)

def select_action(policy, board, player, epsilon=0.0):
    b = board[:] if player==1 else [-v for v in board]
    state = torch.tensor(encode_state(b), dtype=torch.float32).unsqueeze(0)
    logits = policy(state).squeeze(0)
    mask = torch.tensor([0.0 if v==0 else -1e9 for v in b], dtype=torch.float32)
    masked_logits = logits + mask
    
    # ε-random exploration: choose a legal move uniformly; don't backpropagate it
    import random
    if epsilon > 0.0 and random.random() < epsilon:
        moves = [i for i, v in enumerate(b) if v == 0]
        if not moves: 
            return None, None
        a = random.choice(moves)
        return a, None  # <-- DO NOT reinforce random picks

    # From the policy: eval = greedy; training = sample for unbiased PG
    probs = torch.softmax(masked_logits, dim=-1)

    if epsilon == 0.0:
        # evaluation/GUI
        a = torch.argmax(probs).item()
        logp = torch.log(probs[a] + 1e-12)
        return a, logp
    else:
        # training: sample from policy
        m = torch.distributions.Categorical(probs)
        a = m.sample().item()
        logp = m.log_prob(torch.tensor(a))
        return a, logp
