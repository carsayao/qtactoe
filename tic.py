# 0,0 | 0,1 | 0,2
# ————+—————+————
# 1,0 | 1,1 | 1,2
# ————+—————+————
# 2,0 | 2,1 | 2,2

#  0  |  1  |  2
# ————+—————+————
#  3  |  4  |  5
# ————+—————+————
#  6  |  7  |  8


import os
import sys
import pickle
import random
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from subprocess import call

parser = argparse.ArgumentParser(description="Q-Learning Tic-Tac-Toe")
parser.add_argument('--plot', action='store_true',
                    help="plot the results")
parser.add_argument('--debug', action='store_true',
                    help="set trace with pdb")
parser.add_argument('--test', action='store_true',
                    help="test function from test.py")
args = parser.parse_args()
args = vars(args)
PLOT = args["plot"]
if args["debug"] == True:
    import pdb; pdb.set_trace()


class Qmatrix:

    def __init__(self, cols, rows, lr, discount, m, e, delta, agent):
        # plot tools
        self.games = 0
        self.xwins = np.empty(0)
        self.owins = np.empty(0)
        self.draws = np.empty(0)

        self.qmatrix = {}
        self.cols = cols
        self.rows = rows
        self.lr = lr
        self.discount = discount
        self.epochs = m
        self.epsilon = e                    # Decrease by delta every m epochs
        self.epsilon_delta = self.epsilon   # Want to retain original epsilon
        self.delta = delta                  # Value to decrease epsilon
        # TODO: Keep this?
        self.agent = agent                  # Agent is either X (1) or O (-1)
        self.end = False                    # End state
        self.game_state = np.zeros((rows, cols))
        self.save_stats = f"lr{self.lr}-" \
                        + f"e{self.epsilon}-" \
                        + f"d{self.delta}-" \
                        + f"m{self.epochs}-" \
                        + f"g{self.discount}"
        self.file = f"data/policy_x_{self.save_stats}"
    
    def reset(self):
        self.end = False    # End state
        self.game_state = np.zeros((self.rows, self.cols))  # Empty game board

    def show(self, state=None):
        """Pretty print our state"""
        for i in range(len(self.game_state[0])):
            print(str(self.game_state[i])
                      .replace(". ", " | ")
                      .replace('[', ' ')
                      .replace(']', ' ')
                      .replace("'", "")
                      .replace('0', '   ')
                      .replace('.', '')
                      .replace('-1', 'O')
                      .replace('1', 'X')
                      .replace('  ', ' '))
            if i < len(self.game_state[0])-1:
                print("———+———+———")
        # print("show game_state")
        # print(self.game_state)
    
    def actions(self):
        """List available actions from current game board"""
        options = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i][j] == 0:
                    options.append((i,j))
        return options
    
    def choose_action(self, player, play=False):
        """Choose next action to take (epsilon-greedy)

        Args:

            player (int) : 1 for 'X', -1 for 'O'

        Returns:

            action (tuple)
        """
        options = self.actions()
        prob_rand = np.random.uniform(0, 1)
        # If playing, we don't want to randomly select
        epsilon = self.epsilon_delta if play == False else 1
        if prob_rand <= (1 - epsilon):
            action = self.randomly_place(self.agent)
        else:
            # Enumerate all Qs and choose max
            max_action = ()
            max_val = 0
            temp_board = []
            for a in self.actions():
                temp_board = self.encode(self.game_state, player, action=a)
                # Hold onto possible qval
                val = 0 if self.qmatrix.get(temp_board) is None \
                                else self.qmatrix.get(temp_board)
                if val > max_val:
                    max_val = val
                    max_action = a
            action = self.randomly_place(self.agent) if max_action == () \
                          else max_action
        return action

    def randomly_place(self, player):
        """Return randomly selected tuple from legal moves"""
        options = self.actions()
        index = random.choice(options)
        return index
    
    # def encode(self, state, action=None):
    def encode(self, state, player=None, action=None):
        """Return encoded game state to be stored in dictionary.
        
        Args:

            state : (numpy.ndarray) state to encode
            player : (int) optional, 1 for 'X', -1 for 'O'
            action : (int tuple) optional, encode next action to be taken
        
        Returns:

            encoded : (str) encoded state after action (if available)
        """
        copy_state = state.copy()
        if action is None:  # Encode only the current state
            encoded = str(copy_state.reshape(self.rows * self.cols))
        else:   # Place player on board
            assert(player is not None)  # Don't place if no player passed in
            copy_state[action] = player
            encoded = str(copy_state.reshape(self.rows * self.cols))
        return encoded
    
    def update(self, curr_state, next_state, game_status, reward):
        """Update and give reward as necessary"""
        # game_status = self.check_end()
        # print(f"game_status {game_status}")

        # print(f"reward {reward}")
        qcurr = 0 if self.qmatrix.get(curr_state) is None \
                           else self.qmatrix.get(curr_state)
        # print("[!] qcurr", qcurr)
        qnext = 0 if self.qmatrix.get(next_state) is None \
                           else self.qmatrix.get(next_state)
        # print("[!] qnext", qnext)
        self.qmatrix[curr_state] = qcurr + self.lr * \
                              (reward + (self.discount * qnext - qcurr))
    
    def reward(self, states, end_status):
        """Give reward based on end states"""
        assert(end_status != 0)
        if end_status == 1:  # X wins
            reward = 1
        elif end_status == -1: # O wins
            reward = 0
        elif end_status == 2:  # Draw
            reward = 0.5
        self.games += 1
        # backprop by reversing traversed states
        states = states[::-1]
        for i in range(len(states)-1):
            # self.update(states[i], states[i-1], end_status)
            # Since we reversed states, i+1 is our current, i is next
            self.update(states[i+1], states[i], end_status, reward)
            reward = 0 if self.qmatrix.get(states[i+1]) is None \
                           else self.qmatrix.get(states[i+1])
            # print("[!]", "states[curr]", states[i+1], "states[next]", states[i])

    def train(self):
        episode = 1
        # Hold onto agent's states
        agent_states = []
        epoch = 0

        # Train until epsilon is 1
        while self.epsilon_delta <= 1:

            # Observe current (initial, empty) state
            # Set initial state value

            # self.show()

            epoch += 1
            if epoch % self.epochs == 0:
                print(f"\n・・・・・・・・・ EPOCH {epoch}・・・・・・・・・")
                print(f"epsilon {round(self.epsilon_delta, 3)}")
                self.epsilon_delta += self.delta
            # Play a game 
            end_status = 0  # Keep track of end status
            while self.end == False:

                # Choose action a_t that maximizes nex Q val
                # Perform the action
                # print("[#] X Turn")
                # Hold s
                curr_state = self.encode(self.game_state.copy())
                # agent_states.append(curr_state)
                # Hold sₜ₋₁
                next_action = self.choose_action(1)
                next_state = self.encode(self.game_state.copy(), player=1, action=next_action)
                agent_states.append(next_state)
                # Observe the new statue s_t+1
                # self.update(curr_state, next_state, end_status) # Update Q
                # print("[!] Next Action")
                # print(next_action)
                self.game_state[next_action] = 1    # Update game state
                # self.show()

                # If the game is over after the X is placed, place O
                end_status = self.check_end()
                if end_status != 0:
                    # import pdb; pdb.set_trace()
                    # print("[?] END:", self.end)
                    # self.update(curr_state, next_state, end_status)
                    # TODO: Receive reward
                    self.reward(agent_states, end_status)   # After X
                    self.reset()
                    break
                else:
                    # print("[?] END:", self.end)
                    # Place the O
                    # print("[#] O Turn")
                    opponent_action = self.randomly_place(-1)
                    self.game_state[opponent_action] = -1
                    # self.show()
                    # If the game over after O placed, keep playing (Place X)
                    end_status = self.check_end()
                    if end_status != 0:
                        # print("[?] END:", self.end)
                        self.reward(agent_states, end_status)   # After O
                        self.reset()
                        break
                    # print("[?] END:", self.end)
            # End of game
            self.play(10)
            agent_states = []
            end_status = 0
            self.reset()
        # End of epoch
        # print()
    
    def play(self, ngames):
        # print(f"[*] Playing {ngames} times")
        xwins, owins, draws = 0, 0, 0
        for i in range(ngames):
            end_status = 0  # Keep track of end status
            while self.end == False:
                next_action = self.choose_action(1, True)
                self.game_state[next_action] = 1    # Update game state
                # self.show()

                # If the game is over after the X is placed, place O
                end_status = self.check_end()
                if end_status != 0:
                    if end_status == 1:
                        xwins += 1
                    elif end_status == -1:
                        owins += 1
                    elif end_status == 2:
                        draws += 1
                    self.reset()
                    break
                else:
                    opponent_action = self.randomly_place(-1)
                    self.game_state[opponent_action] = -1
                    # self.show()
                    end_status = self.check_end()
                    if end_status != 0:
                        if end_status == 1:
                            xwins += 1
                        elif end_status == -1:
                            owins += 1
                        elif end_status == 2:
                            draws += 1
                        self.reset()
                        break
            self.reset()
        # End of game
        self.xwins = np.append(self.xwins, xwins)
        self.owins = np.append(self.owins, owins)
        self.draws = np.append(self.draws, draws)
    
    def plot(self):
        # Take avg
        ind = round(self.games/50)
        num = self.games
        chop = num if -(num%ind) == 0 else -(num%ind)
        xwins = np.mean(np.array(self.xwins[:chop]).reshape(-1, ind), axis=1)
        owins = np.mean(np.array(self.owins[:chop]).reshape(-1, ind), axis=1)
        draws = np.mean(np.array(self.draws[:chop]).reshape(-1, ind), axis=1)

        ax = plt.subplot(111)
        xaxis = np.arange(1, xwins.shape[0]+1, 1)
        for n in [xwins, owins, draws]:
            plt.scatter(xaxis, n)
        ax.legend(("xwins", "owins", "draws"), loc="lower right")
        title_info = f"Epochs {self.games}; lr {self.lr}; gamma {self.discount}; " \
                   + f"epsilon {self.epsilon}; delta {self.delta}; m {self.epochs} "
        ax.set(xlabel="epoch", ylabel="score/10", title=title_info)
        plt.savefig(f"data/{self.save_stats}.png")    
        plt.close()

    def load(self):
        try:
            with open(self.file, 'rb') as f:
                self.qmatrix = pickle.load(f)
            print("loaded policy")
        except IOError:
            print(f"creating new policy")
    
    def save(self):
        if os.path.isfile(self.file):
            print("qmatrix already exists")
        else:
            print("saving qmatrix")
            with open(self.file, 'wb') as f:
                pickle.dump(self.qmatrix, f, pickle.HIGHEST_PROTOCOL)
    
    def check_end(self):
        """Check winner.

        Returns:

            0 if no winner
            1 if X wins
            -1 if O wins
            2 if draw
        """
        self.delim   = f"=  "*15
        self.we_draw = self.delim + f"DRAW ¯\_(ツ)_/¯ " + self.delim
        self.i_win   = self.delim + f"I Win ᕕ( ᐛ )ᕗ  " + self.delim
        self.you_win = self.delim + f"YOU won (งツ)ว  " + self.delim
        # Check rows
        for i in range(self.rows):
            if np.sum(self.game_state[i]) == 3:
                # print(f"[!] {self.i_win}")
                self.end = True
                return 1
            if np.sum(self.game_state[i]) == -3:
                # print(f"[!] {self.you_win}")
                self.end = True
                return -1
        # Check cols
        for j in range(self.cols):
            if np.sum(self.game_state[:,j]) == 3:
                # print(f"[!] {self.i_win}")
                self.end = True
                return 1
            if np.sum(self.game_state[:,j]) == -3:
                # print(f"[!] {self.you_win}")
                self.end = True
                return -1
        # Check diagonals
        if np.trace(self.game_state) == 3 \
           or np.trace(np.rot90(self.game_state)) == 3:
            # print(f"[!] {self.i_win}")
            self.end = True
            return 1
        if np.trace(self.game_state) == -3 or \
           np.trace(np.rot90(self.game_state)) == -3:
            # print(f"[!] {self.you_win}")
            self.end = True
            return -1
        # Check all states full
        for i in range(self.rows):
            for j in range(self.cols):
                if np.sum(self.game_state[i][j]) == 0:
                    # print("[*] Game still on")
                    return 0
        # print(f"[!] {self.we_draw}")
        self.end = True
        return 2


def main():

    lr = .2
    discount = .9
    m = 5
    epsilon = [0.3, 0.2, 0.1]
    delta = [0.005, 0.003, 0.001, 0.0005, 0.0003, 0.0001]
    agent = 1
    
    print("** init **")
    # cols, rows, lr, discount, epochs, epsilon, delta, agent):
    for e in epsilon:
        for d in delta:
            game = Qmatrix(3, 3, lr, discount, m, e, d, agent)
            game.train()
            game.save()
            game.plot()

    # game.load()
    # game.play(10)
    print("xwins", game.xwins)
    print("owins", game.owins)
    print("draws", game.draws)
    print("len(qmatrix)", len(game.qmatrix))

if __name__ == '__main__':
    main()
