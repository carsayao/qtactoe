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
import random
import argparse
import numpy as np
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

    def __init__(self, cols, rows, ada, gamma, m, e, delta, agent):
        self.qmatrix = {}
        self.cols = cols
        self.rows = rows
        self.ada = ada
        self.gamma = gamma
        self.epochs = m
        self.epsilon = e    # Decrease by delta every m epochs
        self.delta = delta  # Value to decrease epsilon
        self.agent = agent  # Agent is either X (1) or O (-1)
        self.end = False    # End state
        self.game_state = np.zeros((rows, cols))
    
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
        print("show game_state")
        print(self.game_state)
    
    def actions(self):
        """List available actions from current game board"""
        options = []
        for i in range(self.rows):
            for j in range(self.cols):
                if self.game_state[i][j] == 0:
                    options.append((i,j))
        return options
    
    def choose_action(self, player):
        """Choose next action to take (epsilon-greedy)

        Args:

            player (int) : 1 for 'X', -1 for 'O'

        Returns:

            action (tuple)
        """
        options = self.actions()
        print("options",options)
        prob_rand = np.random.uniform(0, 1)
        if .1 <= self.epsilon:
        # if prob_rand <= self.epsilon:
            action = self.randomly_place(self.agent)
        else:
            # Enumerate all Qs and choose max
            print("actions")
            max_action = ()
            max_val = -100
            temp_board = []
            for a in self.actions():
                temp_board = self.encode(self.game_state, player, action=a)

                if self.qmatrix[a] > max_val:
                    max_action = a
            action = max_action
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
        copy_state = self.game_state.copy()
        if action is None:  # Encode only the current state
            encoded = str(copy_state.reshape(self.rows * self.cols))
        else:   # Place player on board
            assert(player is not None)  # Don't place if no player passed in
            copy_state[action] = player
            encoded = str(copy_state.reshape(self.rows * self.cols))
        return encoded
    
    def update(self, curr_state, next_state):
        """Update and give reward as necessary"""
        print("[!] update")
        game_status = self.check_end()
        print(f"game_status {game_status}")
        if game_status == 1:    # X wins
            call(['espeak "x wins, reward 1" 2>/dev/null'], shell=True)
            reward = 1
        if game_status == -1:   # O wins
            call(['espeak "o wins, reward 0" 2>/dev/null'], shell=True)
            reward = 0
        if game_status == 2:    # Draw
            call(['espeak "draw, reward .5" 2>/dev/null'], shell=True)
            reward = .5
        else:
            reward = 0
        print(f"reward {reward}")
        qcurr = 0 if self.qmatrix.get(curr_state) is None \
                           else self.qmatrix.get(curr_state)
        print("[!] qcurr", qcurr)
        qnext = 0 if self.qmatrix.get(next_state) is None \
                           else self.qmatrix.get(next_state)
        print("[!] qnext", qnext)
        self.qmatrix[curr_state] = qcurr + self.ada * \
                              (reward + (self.gamma * qnext - qcurr))
    
    def reward(self):
        """Give reward based on end state"""
        end_state = self.check_end()
        if end_state == 1:  # X wins
            pass
        if end_state == -1: # O wins
            pass
        if end_state == 2:  # Draw
            pass

    def train(self):
        episode = 1
        # Hold onto agent's states
        agent_states = []
        epoch = 0

        # Train until epsilon is 1
        while self.epsilon <= 1:
            print("\nEPOCH", epoch)
            # print("epsilon", self.epsilon)
            # print("agent states\n", agent_states)
            # print("qmatrix", self.qmatrix)
            # Observe current (initial, empty) state
            # Set initial state value
            # self.qmatrix[curr_state] = 0 if self.qmatrix.get(curr_state) is None \
                                            # else self.qmatrix.get(curr_state)
            # print("qmatrix", self.qmatrix)
            self.show()

            epoch += 1
            if epoch % self.epochs == 0:
                print(f"epoch %% self.epochs {epoch, self.epochs}")
                self.epsilon -= self.delta
            # Play a game 
            while self.end == False:
                # print("END?", self.end)
                # print(f"agent states {len(agent_states)}\n{agent_states}")
                # print(f"agent states {len(agent_states)}")#\n{agent_states}")
                # print("qmatrix", self.qmatrix)
                # Choose action a_t that maximizes nex Q val
                # Perform the action
                # Place the X
                print("X Turn")
                # Hold s
                curr_state = self.encode(self.game_state.copy(), player=1)
                agent_states.append(curr_state)
                # Hold sₜ₋₁
                next_action = self.choose_action(1)
                next_state = self.encode(curr_state, player=1, action=next_action)
                # agent_states.append(next_state)
                # Observe the new statue s_t+1
                # Update Q
                # Update game state
                # Receive reward
                self.update(curr_state, next_state)
                self.game_state[next_action] = 1
                self.show()

                # If the game is over after the X is placed, place O
                if self.check_end() != 0:
                    import pdb; pdb.set_trace()
                    print("* * END? * *", self.end)
                    self.update(curr_state, next_state)
                    self.reset()
                    break
                else:
                    print("END?", self.end)
                    print(f"agent states {len(agent_states)}\n{agent_states}")
                    # print(f"agent states {len(agent_states)}")#\n{agent_states}")
                    print("qmatrix", self.qmatrix)

                    # Place the O
                    print("O Turn")
                    opponent_action = self.randomly_place(-1)
                    self.game_state[opponent_action] = -1
                    self.show()

                    # If the game over after O placed, keep playing (Place X)
                    if self.check_end() != 0:
                        import pdb; pdb.set_trace()
                        print("* * END? * *", self.end)
                        self.update(curr_state, next_state)
                        self.reset()
                        break
                    print("END?", self.end)
            self.reset()
    
    def check_end(self):
        """Check winner.

        Returns:

            0 if no winner
            1 if X wins
            -1 if O wins
            2 if draw
        """
        # Check for win
        print("check:")
        # Check rows
        for i in range(self.rows):
            if np.sum(self.game_state[i]) == 3:
                print("X Wins")
                self.end = True
                return 1
            if np.sum(self.game_state[i]) == -3:
                print("O Wins")
                self.end = True
                return -1
        # Check cols
        for j in range(self.cols):
            if np.sum(self.game_state[:,j]) == 3:
                print("X Wins")
                self.end = True
                return 1
            if np.sum(self.game_state[:,j]) == -3:
                print("O Wins")
                self.end = True
                return -1
        # Check diagonals
        if np.trace(self.game_state) == 3 \
           or np.trace(np.rot90(self.game_state)) == 3:
            print("X wins")
            self.end = True
            return 1
        if np.trace(self.game_state) == -3 or \
           np.trace(np.rot90(self.game_state)) == -3:
            print("O wins")
            self.end = True
            return -1
        # Check all states full
        for i in range(self.rows):
            for j in range(self.cols):
                if np.sum(self.game_state[i][j]) == 0:
                    print("False")
                    return 0
        print("Draw")
        self.end = True
        return 2


def main():
    game = Qmatrix()

if __name__ == '__main__':
    main()
