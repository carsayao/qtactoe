import numpy as np
import random


class Simple:
    def __init__(self, cols, rows, ada, m, e, delta):
        self.qmatrix = {}
        self.cols = cols
        self.rows = rows
        self.ada = ada
        self.epochs = m
        self.epsilon = e    # Decrease by delta every m epochs
        self.delta = delta  # Value to decrease epsilon
        self.end = False    # End state
        self.game_state = np.zeros(cols)
        # self.game_state = np.zeros((rows, cols))
        self.agent = ()
        # index = self.game_state.shape[1]

    def show(self, action):
        print(str(self.game_state)
                  .replace("[", "")
                  .replace("]", "")
                  .replace("0.", "0")
                  .replace("1", "A")
                  .replace(".", "")
        )
    
    def actions(self):
        options = []
        # print('a', self.agent[0], self.agent[1])
        if self.agent-1 >= 0:
            options.append(self.agent-1)
        if self.agent+1 < 4:
            options.append(self.agent+1)
        # if self.agent[1]-1 >= 0:
        #     options.append(self.agent[1]-1)
        # if self.agent[1]+1 < 4:
        #     options.append(self.agent[1]+1)
        return options
    
    def choose_action(self):
        if np.random.uniform(0,1) <= self.epsilon:
            action = self.randomly_place(self.agent)
        else:
            pass
        return action
    
    def randomly_place(self, agent):
        options = self.actions()
        index = random.choice(options)
        return index
    
    def play(self):
        self.show((0,0))
            # Hold onto agent's states
        agent_states = []
        # Observe current state
        curr_state = self.game_state.copy()
        self.agent = 2  # Current state
        # self.agent = (0,2)  # Current state
        self.game_state[self.agent] = 1
        # self.game_state[0][self.agent[1]] = 1
        # Choose action a_t
        print("actions", self.actions())
        print("choice", self.randomly_place(self.agent))
        # Perform the action
        next_state = self.encode(curr_state, self.choose_action())
        # Observe the new statue s_t+1
        # print(self.qmatrix[next_state] is None)
        reward = 0 if self.qmatrix.get(next_state) is None else qmatrix.get(next_state)
        # Update Q
        # Receive reward
        print(reward)

    def encode(self, state, action):
        """Return encoded game state to be stored in dictionary"""
        encoded = str(np.append(self.game_state.reshape(self.rows * self.cols), action))
        return encoded
    

def main():

    game = Simple(4, 1, 1, .1, 5, .05)
    game.play()

if __name__ == "__main__":
    main()
