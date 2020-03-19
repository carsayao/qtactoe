import os
import sys
import pickle
# sys.path.append('../')
from tic import Qmatrix


try:
    arg = sys.argv[2]
except IndexError:
    print(f"You need an arg")

# with open('policy_x', 'rb') as f:
    # qmatrix = pickle.load(f)
import pdb; pdb.set_trace()


lr = 1
discount = .9
m = 5
e = 0.1
# ~ 89 epochs
# delta = 0.0005
delta = 1
agent = 1
sys.exit()

print("** init **")
# cols, rows, lr, discount, epochs, epsilon, delta, agent):
game = Qmatrix(3, 3, lr, discount, m, e, delta, agent)
game.train()

if arg == 'reward':
    game.reward()

sys.exit()
