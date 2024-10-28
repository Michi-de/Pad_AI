# agent.py

import torch
import random
import numpy as np
from collections import deque
from model import Linear_QNet, QTrainer
from Game import PongGameAI

# Set device to MPS if available, otherwise use CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Constants
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(4, 1028, 3).to(device)  # Move model to MPS/CPU after instantiation
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return np.array(game.get_state(), dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    game = PongGameAI()
    agent = Agent()
    record = 0

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, next_state = game.play_step(final_move.index(1))
        
        # Train short memory after each action
        agent.train_short_memory(state_old, final_move, reward, next_state, done)
        agent.remember(state_old, final_move, reward, next_state, done)

        if done:
            game.reset_ball()
            agent.n_games += 1
            agent.train_long_memory()

            if game.AI_score > record:
                record = game.AI_score

            print(f'Game {agent.n_games} - Score: {game.AI_score} - Record: {record}')

if __name__ == "__main__":
    train()