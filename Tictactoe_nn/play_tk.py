import tkinter as tk
from tkinter import messagebox
import numpy as np, torch
from core import PolicyNet, check_winner, legal_moves, heuristic_move, select_action

class App:
    def __init__(self, root):
        self.root=root
        self.root.title('Neural Tic-Tac-Toe')
        self.board=[0]*9
        self.btns=[]
        self.diff=tk.StringVar(value='Medium')
        self.status=tk.StringVar(value='Your turn (X). Difficulty: Medium')
        self.model=PolicyNet()
        try: 
            self.model.load_state_dict(torch.load('ttt_model.pt', map_location='cpu'))
        except Exception: pass
        self.model.eval()
        top=tk.Frame(root)
        top.pack(pady=6)
        for i in range(3):
            fr=tk.Frame(top)
            fr.pack()
            row=[]
            for j in range(3):
                idx=i*3+j
                btn=tk.Button(fr, text=' ', width=6, height=3, font=('Arial',18),
                              command=lambda k=idx: self.human(k))
                btn.grid(row=i, column=j, padx=3, pady=3); row.append(btn)
            self.btns.append(row)
        ctrl=tk.Frame(root)
        ctrl.pack(pady=8)
        tk.Radiobutton(ctrl, text='Easy (Random)', variable=self.diff, value='Easy', command=self.refresh).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(ctrl, text='Medium (Heuristic)', variable=self.diff, value='Medium', command=self.refresh).pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(ctrl, text='Hard (Neural Net)', variable=self.diff, value='Hard', command=self.refresh).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='New Game', command=self.reset).pack(side=tk.LEFT, padx=10)
        tk.Label(root, textvariable=self.status, font=('Arial',12)).pack(pady=4)

    def refresh(self): 
        self.status.set(f'Your turn (X). Difficulty: {self.diff.get()}')

    def reset(self):
        self.board=[0]*9
        for i in range(3):
            for j in range(3):
                self.btns[i][j]['text']=' '
                self.btns[i][j]['state']=tk.NORMAL
        self.refresh()

    def human(self, idx):
        if self.board[idx]!=0: 
            return
        self.board[idx]=1
        self.btns[idx//3][idx%3]['text']='X'
        self.after_move()

    def after_move(self):
        res=check_winner(self.board)
        if res is not None: 
            return self.finish(res)
        if self.diff.get()=='Easy':
            mv=legal_moves(self.board)
            a=None if not mv else np.random.choice(mv)
        elif self.diff.get()=='Medium':
            a=heuristic_move(self.board, player=-1)
        else:
            a,_=select_action(self.model, self.board, player=-1, epsilon=0.0)
        if a is None: 
            return self.finish(0)
        self.board[a]=-1
        self.btns[a//3][a%3]['text']='O'
        res=check_winner(self.board)
        if res is not None: 
            return self.finish(res)

    def finish(self, res):
        for i in range(3):
            for j in range(3):
                self.btns[i][j]['state']=tk.DISABLED
        if res==1: messagebox.showinfo('Game Over','You win! 🎉')
        elif res==-1: messagebox.showinfo('Game Over','AI wins! 🤖')
        else: messagebox.showinfo('Game Over','Draw.')

if __name__=='__main__':
    root=tk.Tk()
    App(root)
    root.mainloop()
