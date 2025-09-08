# Neural Network Tic-Tac-Toe (PyTorch)

This package includes:
- `core.py` — board logic + policy + heuristic/random opponents
- `train.py` — REINFORCE trainer (random / heuristic / self-play)
- `play_tk.py` — Tkinter GUI with Easy/Medium/Hard
- `ttt_model.pt` — starter weights (train more for a stronger Hard)

## Run
`python play_tk.py`

## Train
```
python train.py --opponent random    --episodes 2000 --out ttt_model.pt
python train.py --opponent heuristic --episodes 3000 --out ttt_model.pt
python train.py --opponent self      --episodes 2000 --snapshot_every 200 --out ttt_model.pt
```

