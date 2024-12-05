
from random import randrange, randint
import numpy as np
import pandas as pd

from DataFetcher import *
class State:
    def __init__(self, name, reward, songs, state_probs):
        """
        :param name: The state name
        :param reward: An instance of Reward class
        :param songs: An array of instances of the Song class
        :param state_probs: Estimated probabilities of moving to neighbouring states
        :param neigh_states: Instances of neighbouring states
        """
        self.name = name
        self.reward = reward.calculate_reward()
        self.songs = songs
        self.alpha = state_probs
        self.neighbours = []
        self.actions = []
        self.transition_probabilities = []

    def set_alpha_values(self, input_values):
        self.alpha = input_values
        self.transition_probabilities = np.random.dirichlet(self.alpha)

    def set_reward(self, rew):
        self.reward = rew
        self.transition_probabilities = np.random.dirichlet(self.alpha)  # Recalculate transition probabilities
        if sum(self.transition_probabilities) != 1:
            to_add = 1 - sum(self.transition_probabilities)
            self.transition_probabilities[randint(0, len(self.transition_probabilities) - 1)] += to_add

    def add_neighbour(self, state):
        self.neighbours.append(state)
        self.alpha.append(randrange(10, 100))
        self.actions.append('Consider recommending the state ' + str(state.name))
        self.transition_probabilities = np.random.dirichlet(self.alpha)  # ?
        if (sum(self.transition_probabilities)) != 1:
            to_add = 1 - sum(self.transition_probabilities)
            self.transition_probabilities[randint(0, len(self.transition_probabilities) - 1)] += to_add

    def move_to_state(self, state):
        idx = self.neighbours.index(state)
        self.alpha[idx] += 1
        self.transition_probabilities = np.random.dirichlet(self.alpha)  # ?
        if (sum(self.transition_probabilities)) != 1:
            to_add = 1 - sum(self.transition_probabilities)
            self.transition_probabilities[randint(0, len(self.transition_probabilities))] += to_add

    def transition_probability_to(self, state):
        idx = self.neighbours.index(state)
        return self.transition_probabilities[idx]

    def get_state(self):
        return self.name + "\n" + str(round(self.reward, 3))

    def get_neighbours(self):
        return self.neighbours

    # Best for testing purposes
    def print_information(self):
        print("------------------------------------------------------------------------")
        print(self.name.upper())
        max_price, max_neigh, flag = 0, None, 0
        for neighbour in self.neighbours:
            reward = neighbour.reward
            print("Neighbour: ", neighbour.name)
            print("     Neighbour reward: ", reward)
            if flag == 0:
                max_price = reward
                max_neigh = neighbour
                flag = 1
            if reward > max_price:
                max_price = reward
                max_neigh = neighbour

        print("++++++ Best reward has the neighbour: ", max_neigh.name, " with reward: ", max_price)

    def random_rec(self):
        if len(self.neighbours) < 5:
            index = self.songs[-1].get_most_similar(dataset)
            song = dataset[index]
            song_name = song[-2]
            artist = song[-1]
            print("The user is listening to ", self.songs[-1].song_information[-2], " by ",
                  self.songs[-1].song_information[-1])
            print('Consider recommending: ', song_name, " by ", artist)