import argparse, json, random
import torch, torch.optim as optim
import matplotlib.pyplot as plt
import os
from core import PolicyNet, select_action, heuristic_move, check_winner

def legal_moves(board): return [i for i,v in enumerate(board) if v==0]

def play_episode(policy, opponent='random', snapshot=None, epsilon=0.1):
    board=[0]*9
    player=1
    log_probs=[]
    while True:
        term = check_winner(board)
        if term is not None:
            reward = 1.0 if term==1 else (-1.0 if term==-1 else 0.0)
            return reward, log_probs
        if player==1:
            a, logp = select_action(policy, board, player=1, epsilon=epsilon)
            if a is None: 
                return 0.0, log_probs
            board[a]=1
            if logp is not None: 
                log_probs.append(logp)
        else:
            if opponent=='random':
                moves = legal_moves(board)
                a = None if not moves else random.choice(moves)
            elif opponent=='heuristic':
                a = heuristic_move(board, player=-1)
            elif opponent=='self':
                a, _ = select_action(snapshot, board, player=-1, epsilon=0.0)
            else:
                moves = legal_moves(board)
                a = None if not moves else random.choice(moves)
            if a is None: 
                return 0.0, log_probs
            board[a] = -1
        player *= -1

def train(episodes=2000, opponent='random', lr=1e-3, epsilon=0.1, snapshot_every=200, log_every=50, window=100, 
          seed=0, out='ttt_model.pt'):
    random.seed(seed)
    torch.manual_seed(seed)
    policy = PolicyNet()
    if os.path.exists(out):
        try:
            policy.load_state_dict(torch.load(out, map_location="cpu"))
            print(json.dumps({"resumed_from": out}))
        except Exception as e:
            print(json.dumps({"resume_failed": str(e)}))
    opt = optim.Adam(policy.parameters(), lr=lr)
    baseline=0.0
    beta=0.99
    snapshot = PolicyNet()
    snapshot.load_state_dict(policy.state_dict())
    snapshot.eval()
    losses=[]
    rewards=[]
    for ep in range(1, episodes+1):
        if opponent=='self' and ep % snapshot_every == 1 and ep>1:
            snapshot.load_state_dict(policy.state_dict())
            snapshot.eval()

        if opponent == 'self':
            opp_now = 'self' if random.random() < 0.7 else 'heuristic'
        else:
            opp_now = opponent

        # ε-decay: explore early, exploit later
        min_eps = 0.02
        eps_now = max(min_eps, epsilon * (1.0 - ep / episodes))

        r, logs = play_episode(policy, opponent=opp_now, snapshot=snapshot, epsilon=eps_now)

        # moving-average baseline
        baseline = beta * baseline + (1 - beta) * r
        adv = r - baseline

        if logs:
            loss = -torch.stack(logs).sum() * adv
            opt.zero_grad()
            loss.backward()
            # prevent rare gradient explosions
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        else:
            loss = torch.tensor(0.0)
        losses.append(loss.item())
        rewards.append(r)
        recent = rewards[-window:]
        winrate = sum(1 for x in recent if x > 0) / max(1, len(recent))

        if ep % log_every == 0 or ep == 1:
            print(json.dumps({
                "episode": ep,
                "loss": float(loss.item()),
                "winrate": winrate
            }))

    torch.save(policy.state_dict(), out)
    print(json.dumps({'out': out, 'episodes': episodes}))

    wr_list = []
    for i in range(episodes):
        sub = rewards[max(0, i-window+1):i+1]
        wr_list.append(sum(1 for x in sub if x>0)/max(1,len(sub)))

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (REINFORCE)")
    plt.xlabel("Episode"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(wr_list)
    plt.title(f"Rolling Win Rate (window={window})")
    plt.xlabel("Episode"); plt.ylabel("Win Rate"); plt.tight_layout()
    plt.savefig("winrate_curve.png")
    plt.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=2000)
    p.add_argument('--opponent', choices=['random','heuristic','self'], default='random')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epsilon', type=float, default=0.1)
    p.add_argument('--snapshot_every', type=int, default=200)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='ttt_model.pt')
    a=p.parse_args()
    train(episodes=a.episodes, opponent=a.opponent, lr=a.lr, epsilon=a.epsilon,
          snapshot_every=a.snapshot_every, log_every=a.log_every, window=a.window, 
          seed=a.seed, out=a.out)
