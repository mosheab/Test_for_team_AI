import argparse, json, random
import torch, torch.optim as optim
import matplotlib.pyplot as plt
import os
from core import PolicyNet, select_action, heuristic_move, check_winner, WIN_LINES

def legal_moves(board): return [i for i,v in enumerate(board) if v==0]


def has_two_in_row_threat(board, player):
    for a,b,c in WIN_LINES:
        line = [board[a], board[b], board[c]]
        if line.count(player) == 2 and line.count(0) == 1:
            return True
    return False

def opponent_can_win_next(board, player):
    opp = -player
    for m in legal_moves(board):
        b2 = board[:]
        b2[m] = opp
        if check_winner(b2) == opp:
            return True
    return False

def play_episode(policy, opponent='random', snapshot=None, epsilon=0.1, agent_player=1):
    board = [0]*9; 
    log_probs = []
    shaped_total = 0.0
    to_move = 1
    while True:
        term = check_winner(board)
        if term is not None:
            reward = 1.0 if term==agent_player else (-1.0 if term==-agent_player else 0.0)
            # small penalty on losing terminal
            if term == -agent_player:
                shaped_total += -0.1
            return reward, shaped_total, log_probs
 
        if to_move == agent_player:
            a, logp = select_action(policy, board, player=agent_player, epsilon=epsilon)
            if a is None: return 0.0, log_probs
            board[a] = agent_player
            if logp is not None:
                log_probs.append(logp)
            # +0.2 if agent created a two-in-a-row threat
            if has_two_in_row_threat(board, agent_player):
                shaped_total += 0.2
            # -0.2 if opponent now has an immediate winning reply
            if opponent_can_win_next(board, agent_player):
                shaped_total += -0.2
            # +0.1 for staying alive (game not ended yet)
            if check_winner(board) is None:
                shaped_total += 0.1
        else:
            if opponent=='random':
                moves = legal_moves(board)
                a = None if not moves else random.choice(moves)
            elif opponent=='heuristic':
                a = heuristic_move(board, player=to_move)
            else:
                a, _ = select_action(snapshot, board, player=to_move, epsilon=0.0)
            if a is None: return 0.0, 0.0, log_probs
            board[a] = to_move
        to_move *= -1


def train(episodes=10000, opponent='random', lr=1e-3, epsilon=1.0,
          snapshot_every=200, log_every=200, window=200, seed=0, out='ttt_model.pt'):
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

    baseline = {1: 0.0, -1: 0.0}
    beta = 0.9

    snapshot = PolicyNet()
    snapshot.load_state_dict(policy.state_dict())
    snapshot.eval()

    losses, rewards = [], []
    for ep in range(1, episodes+1):
        if opponent=='self' and ep % snapshot_every == 1 and ep > 1:
            snapshot.load_state_dict(policy.state_dict())
            snapshot.eval()

        # ε-decay: start random (high temp), cool down
        min_eps = 0.1
        eps_now = max(min_eps, epsilon * (1.0 - ep / episodes))
        agent_player = 1 if random.random() < 0.5 else -1

        r_raw, r_shaped, logs = play_episode(policy, opponent=opponent, snapshot=snapshot,
                               epsilon=eps_now, agent_player=agent_player)

        # Use shaped reward for learning signal
        baseline[agent_player] = beta * baseline[agent_player] + (1 - beta) * r_shaped
        adv = r_shaped - baseline[agent_player]

        if logs:
            loss = -torch.stack(logs).sum() * adv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        else:
            loss = torch.tensor(0.0)

        losses.append(loss.item())
        # Keep raw outcomes for accurate draw-rate reporting
        rewards.append(r_raw)

        if ep % log_every == 0 or ep == 1:
            recent = rewards[-window:]
            draw_rate = sum(1 for x in recent if x == 0) / max(1, len(recent))
            print(json.dumps({
                "episode": ep,
                "loss": float(loss.item()),
                "draw_rate": draw_rate
            }))

    torch.save(policy.state_dict(), out)
    print(json.dumps({'out': out, 'episodes': episodes}))
                     

    # optimal game finishes with a draw
    draw_list = []
    for i in range(episodes):
        sub = rewards[max(0, i-window+1):i+1]
        draw_list.append(sum(1 for x in sub if x==0)/max(1,len(sub)))

    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss (REINFORCE)")
    plt.xlabel("Episode"); plt.ylabel("Loss"); plt.tight_layout()
    plt.savefig("loss_curve.png")
    plt.close()

    plt.figure()
    plt.plot(draw_list)
    plt.title(f"Rolling Draw Rate (window={window})")
    plt.xlabel("Episode"); plt.ylabel("Draw Rate"); plt.tight_layout()
    plt.savefig("drawrate_curve.png")
    plt.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--opponent', choices=['random','heuristic','self'], default='self')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--epsilon', type=float, default=1)
    p.add_argument('--snapshot_every', type=int, default=200)
    p.add_argument('--log_every', type=int, default=50)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=str, default='ttt_model.pt')
    a=p.parse_args()
    train(episodes=a.episodes, opponent=a.opponent, lr=a.lr, epsilon=a.epsilon,
          snapshot_every=a.snapshot_every, log_every=a.log_every, window=a.window, 
          seed=a.seed, out=a.out)
