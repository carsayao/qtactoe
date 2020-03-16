import os
import sys
# sys.path.append('../')
from tic import Qmatrix


try:
    arg = sys.argv[2]
except IndexError:
    print(f"You need an arg")

ada = 1
gamma = .9
m = 5
e = 0.1
delta = -0.5
agent = 1

print("** init **")
game = Qmatrix(3, 3, ada, gamma, m, e, delta, agent)
# game.show()
# end = game.check_end()
# print("end", end)
# print("rand", game.choose_action())
# end = 2

game.train()

# while end == 0:
#     game.randomly_place(1)
#     print("end1", end)
#     game.show()
#     end = game.check_end()

#     print("end2", end)
#     if end == 0:
#         game.randomly_place(-1)
#     game.show()
#     end = game.check_end()
#     print("end3", end)

# game.show()
# game.check_end()
# game.randomly_place(1)

sys.exit()
