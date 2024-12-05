# import csv
# import random
# from random import randrange, random, randint
# import networkx as nx
#
# import matplotlib.pyplot as plt
#
# from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
# from mdptoolbox import mdp
#
# kNN_classifier = KNeighborsClassifier(n_neighbors=5)
#
#
# # Helper functions:
#
#
# def read_file(file_name):
#     with open(file_name, newline='', encoding='utf-8') as doc:
#         csv_reader = csv.reader(doc, delimiter=',')
#         dataset = list(csv_reader)[1:]
#
#         dataset_v2 = []
#         for row in dataset:
#             transformed = []
#             for element in row[1:]:
#                 try:
#                     transformed.append(float(element))
#                 except ValueError:
#                     transformed.append(element)
#
#             dataset_v2.append(transformed)
#
#     return dataset_v2
#
#
# def get_attributes(path):
#     with open(path, newline='', encoding='utf-8') as doc:
#         csv_reader = csv.reader(doc, delimiter=',')
#         attributes = list(csv_reader)[0][1:]
#     return attributes
#
#
# # DATASET AND ATTRIBUTES
# dataset_path = "C:\\Users\\janaa\\PycharmProjects\\MDP recommendation system\\Dataset\\data.csv"
# dataset = read_file(dataset_path)
# attributes = get_attributes(dataset_path)
#
# import numpy as np
#
#
# class MDPSolver:
#     def __init__(self, states, gamma=0.9, threshold=0.001):
#         """
#         :param states: A dictionary of states, where keys are state names and values are State instances
#         :param gamma: Discount factor for future rewards
#         :param threshold: Convergence threshold for value iteration
#         """
#         self.states = states
#         self.gamma = gamma
#         self.threshold = threshold
#         self.state_values = {state.name: 0 for state in states.values()}
#         self.iter = 0
#         self.epsilon = 10
#
#     def value_iteration(self):
#         """
#         Perform value iteration to find the optimal state values
#         """
#         self.iter += 1
#         if self.iter > 400:
#             self.epsilon = self.epsilon / 10
#
#         while True:
#             delta = 0
#             for state in self.states.values():
#
#                 # allow exploration with epsilon greedy
#                 if np.random.random() < self.epsilon:
#                     return state.neighbours[randrange(0, len(state.neighbours) - 1)]
#                 max_value = float('-inf')
#                 prob = 0
#                 for neighbor in state.neighbours:
#                     transition_prob = state.transition_probability_to(neighbor)
#
#                     reward = state.reward
#                     # prob += transition_prob # for testing purposes
#                     next_value = reward + self.gamma * self.state_values[neighbor.name]
#                     max_value = max(max_value, transition_prob * next_value)
#                     print("Max value for state ", state.name, " is ", max_value)
#                 # print("PROBABILITY CHECK --------------------------", prob)
#
#                 delta = max(delta, abs(max_value - self.state_values[state.name]))
#                 self.state_values[state.name] = max_value
#
#             if delta < self.threshold:
#                 break
#
#         return self.state_values
#
#     def get_optimal_policy(self):
#         """
#         Derive the optimal policy based on the computed state values
#         """
#         policy = {}
#         for state in self.states.values():
#             best_action = None
#             best_value = float('-inf')
#             prob = 0
#             for neighbor in state.neighbours:
#                 transition_prob = state.transition_probability_to(neighbor)
#                 prob += transition_prob
#                 expected_value = transition_prob * (state.reward + self.gamma * self.state_values[neighbor.name])
#                 if expected_value > best_value:
#                     best_value = expected_value
#                     best_action = neighbor
#
#             policy[state.name] = best_action.name if best_action else None
#
#         return policy
#
#
# class State:
#     def __init__(self, name, reward, songs, state_probs):
#         """
#         :param name: The state name
#         :param reward: An instance of Reward class
#         :param songs: An array of instances of the Song class
#         :param state_probs: Estimated probabilities of moving to neighbouring states
#         :param neigh_states: Instances of neighbouring states
#         """
#         self.name = name
#         self.reward = reward.calculate_reward()
#         self.songs = songs
#         self.alpha = state_probs
#         self.neighbours = []
#         self.actions = []
#         self.transition_probabilities = []
#
#     def set_alpha_values(self, input_values):
#         self.alpha = input_values
#         self.transition_probabilities = np.random.dirichlet(self.alpha)
#
#     def set_reward(self, rew):
#         self.reward = rew
#         self.transition_probabilities = np.random.dirichlet(self.alpha)  # Recalculate transition probabilities
#         if sum(self.transition_probabilities) != 1:
#             to_add = 1 - sum(self.transition_probabilities)
#             self.transition_probabilities[randint(0, len(self.transition_probabilities) - 1)] += to_add
#
#     def add_neighbour(self, state):
#         self.neighbours.append(state)
#         self.alpha.append(randrange(10, 100))
#         self.actions.append('Consider recommending the state ' + str(state.name))
#         self.transition_probabilities = np.random.dirichlet(self.alpha)  # ?
#         if (sum(self.transition_probabilities)) != 1:
#             to_add = 1 - sum(self.transition_probabilities)
#             self.transition_probabilities[randint(0, len(self.transition_probabilities) - 1)] += to_add
#
#     def move_to_state(self, state):
#         idx = self.neighbours.index(state)
#         self.alpha[idx] += 1
#         self.transition_probabilities = np.random.dirichlet(self.alpha)  # ?
#         if (sum(self.transition_probabilities)) != 1:
#             to_add = 1 - sum(self.transition_probabilities)
#             self.transition_probabilities[randint(0, len(self.transition_probabilities))] += to_add
#
#     def transition_probability_to(self, state):
#         idx = self.neighbours.index(state)
#         return self.transition_probabilities[idx]
#
#     def get_state(self):
#         return self.name + "\n" + str(round(self.reward, 3))
#
#     def get_neighbours(self):
#         return self.neighbours
#
#     def print_information(self):
#         print("------------------------------------------------------------------------")
#         print(self.name.upper())
#         max_price, max_neigh, flag = 0, None, 0
#         for neighbour in self.neighbours:
#             reward = neighbour.reward
#             print("Neighbour: ", neighbour.name)
#             print("     Neighbour reward: ", reward)
#             if flag == 0:
#                 max_price = reward
#                 max_neigh = neighbour
#                 flag = 1
#             if reward > max_price:
#                 max_price = reward
#                 max_neigh = neighbour
#
#         print("++++++ Best reward has the neighbour: ", max_neigh.name, " with reward: ", max_price)
#
#     def random_rec(self):
#         if len(self.neighbours) < 5:
#             index = self.songs[-1].get_most_similar(dataset)
#             song = dataset[index]
#             song_name = song[-2]
#             artist = song[-1]
#             print("The user is listening to ", self.songs[-1].song_information[-2], " by ",
#                   self.songs[-1].song_information[-1])
#             print('Consider recommending: ', song_name, " by ", artist)
#
#
# class Song:
#     def __init__(self, attributes, dataset_information):
#         self.attributes = attributes
#         self.song_information = dataset_information
#         self.data_pairs = {}
#         self.name = dataset_information[-1]
#         self.artist = dataset_information[-2]
#         # ATTRIBUTE NAMES
#         # acousticness,danceability,duration_ms,energy,instrumentalness,
#         # key,liveness,loudness,mode,speechiness,tempo,time_signature,valence,target,song_title,artist
#
#         # Map for attributes and their value
#         for attr, song_info in zip(self.attributes, self.song_information):
#             self.data_pairs[attr] = song_info
#
#     def get_most_similar(self, all_songs_attributes):
#         """
#         Function to find the most similar songs to the current song, using the k-NN algorithm.
#         :param all_songs_attributes: A 2D list or array of features for all songs in the dataset.
#         :return: The index of the most similar song.
#         """
#         print(self.song_information)
#         knn = NearestNeighbors(n_neighbors=2, metric='cosine')
#         # exclude the current song from the database
#         all_songs_attributes = [song[:-2] for song in all_songs_attributes if song != self.song_information]
#         knn.fit(all_songs_attributes)
#         # print("KNN FITTED")
#         current_song_features = [self.song_information[:-2]]
#         # print("FEATURES EXTRACTED FROM CURRENT SONG")
#         distances, indices = knn.kneighbors(current_song_features)
#         # print(indices)
#         return indices[0][0]
#
#     def name_and_artist(self):
#         print(self.name, "\n", self.artist)
#
#     def find_similarity(self, songs):
#         avg_sim = 0
#         for song in songs:
#             info = np.array(song.database_information)
#             this_song = np.array(self.song_information)
#             avg_sim += np.linalg.norm(info - this_song)
#         return avg_sim / (len(songs) + 1)
#
#
# def simulate(good_map=None, bad_map=None):
#     """
#     This function simulates random user actions while listening to ONE song, such that typical user actions are divided
#     into 2 categories: positive and negative. The user will perform one or the other based on a single random decision.
#     :return: Values for good and bad actions, possible rating and the name of the user
#     """
#     gave_rating = -1
#
#     good_actions_map = {
#         'repeat': False,
#         'added_song': False,
#         'shared': False,
#     }
#     bad_actions_map = {
#         'skip': False,
#         'removed_song': False
#     }
#     # random decision for whether the listener performs positive or negative actions
#
#     if good_map is not None:
#         good_actions_map = good_map
#     if bad_map is not None:
#         bad_actions_map = bad_map
#
#     listening_percentage = random()
#     if listening_percentage < 0.05:
#         bad_actions_map['skip'] = True
#
#     if listening_percentage > 0.4:
#         for action in good_actions_map.keys():
#             if random() > 0.8:
#                 good_actions_map[action] = True
#         if random() > 0.8:
#             gave_rating = randrange(3, 5)
#     else:
#         if random() > 0.5:
#             bad_actions_map['removed_song'] = True
#         if random() > 0.8:
#             gave_rating = randrange(0, 2)
#
#     return good_actions_map, bad_actions_map, listening_percentage, gave_rating
#
#
# class Reward:
#     def __init__(self, current_song, previous_songs):
#         """
#         Reward class for forming and calculating the reward AFTER the used left the state (switched songs)
#         :param current_song: An instance of the currently playing song
#         :param previous_songs: An instance of the last two songs played
#         """
#         self.current_song = current_song
#         self.last_songs = previous_songs
#
#     def calculate_reward(self, simulation=None):
#
#         if simulation is not None:
#             good, bad, listening_time, gave_rating = simulation
#         else:
#             good, bad, listening_time, gave_rating = simulate()
#         # print(good, " - GOOD") # testing
#         # print(bad, " - BAD")
#         # print(listening_time, " - LISTENING TIME")
#         # print(gave_rating, " - GAVE RATING")
#         #  Negative reward for listening for a short time
#         if listening_time < 0.025:
#             listening_time = listening_time * (-4)
#         elif 0.025 < listening_time < 0.25:
#             listening_time = listening_time * (-2)
#
#         positive = 0
#         negative = 0
#         for key in good.keys():
#             if good[key]:
#                 positive += 1
#
#         for key in bad.keys():
#             if bad[key]:
#                 negative -= 1
#         if simulation is None:
#             if 0 <= gave_rating <= 2:
#                 negative -= 2
#             if 4 <= gave_rating <= 5:
#                 positive += 1
#
#             if positive == 0 and negative == 0:
#                 listening_time = listening_time / 2
#
#         return (listening_time + positive + negative) / 4
#
#
# if __name__ == '__main__':
#     print("Enter the number of the states")
#     # state_num = int(input())
#     # # generate names for each state
#     # state_names = []
#     # for i in range(state_num):
#     #     state_names.append("State " + str(i))
#     #
#     # # Create instances of each song from the database
#     # all_songs = []
#     # for entry in dataset:
#     #     song = Song(attributes, entry)
#     #     all_songs.append(song)
#     #
#     # # Generated reward for every state
#     # rewards = []
#     # last_songs = []
#     # for i in range(len(state_names)):
#     #     prev_songs = [all_songs[randint(0, len(all_songs) - 1)], all_songs[randint(0, len(all_songs) - 1)]]
#     #     last_songs.append(prev_songs)
#     #     reward = Reward(all_songs[randint(0, len(all_songs) - 1)], prev_songs)
#     #     rewards.append(reward)
#     #
#     # # name to state dictionary
#     # states = {}
#     # for i in range(len(state_names)):
#     #     # at the beginning, the state probabilities array is empty, since we have 0 neighbours for each state
#     #     state = State(state_names[i], rewards[i], last_songs[i], [])
#     #     states[state_names[i]] = state
#     #
#     # neighbour_states = []
#     # for i in range(state_num):
#     #     neighbours = []
#     #     current_state = "State " + str(i)
#     #     while True:
#     #         generated = randint(0, len(state_names) - 1)
#     #         name = "State " + str(generated)
#     #         if name not in neighbours and name != current_state:  # make sure that we don't have duplicate neighbours of a state
#     #             neighbours.append(name)
#     #             states[current_state].add_neighbour(states[name])
#     #             if len(neighbours) == int(state_num / 2):
#     #                 break
#     #
#     #     neighbour_states.append(neighbours)
#     #
#     # solver = MDP(states.values())
#     # solver.solve()
#     # solver.display_results()
