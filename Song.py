import numpy as np
from sklearn.neighbors import NearestNeighbors
from DataFetcher import *

class Song:
    def __init__(self, attributes, dataset_information):
        self.attributes = attributes
        self.song_information = dataset_information
        self.data_pairs = {}
        self.name = dataset_information[-1]
        self.artist = dataset_information[-2]
        # ATTRIBUTE NAMES
        # acousticness,danceability,duration_ms,energy,instrumentalness,
        # key,liveness,loudness,mode,speechiness,tempo,time_signature,valence,target,song_title,artist

        # Map for attributes and their value
        for attr, song_info in zip(self.attributes, self.song_information):
            self.data_pairs[attr] = song_info

    def get_most_similar(self, all_songs_attributes):
        """
        Function to find the most similar songs to the current song, using the k-NN algorithm.
        :param all_songs_attributes: A 2D list or array of features for all songs in the dataset.
        :return: The index of the most similar song.
        """
        print(self.song_information)
        knn = NearestNeighbors(n_neighbors=2, metric='cosine')
        # exclude the current song from the database
        all_songs_attributes = [song[:-2] for song in all_songs_attributes if song != self.song_information]
        knn.fit(all_songs_attributes)
        # print("KNN FITTED")
        current_song_features = [self.song_information[:-2]]
        # print("FEATURES EXTRACTED FROM CURRENT SONG")
        distances, indices = knn.kneighbors(current_song_features)
        # print(indices)
        return indices[0][0]

    def name_and_artist(self):
        print(self.name, "\n", self.artist)

    def find_similarity(self, songs):
        avg_sim = 0
        for song in songs:
            info = np.array(song.database_information)
            this_song = np.array(self.song_information)
            avg_sim += np.linalg.norm(info - this_song)
        return avg_sim / (len(songs) + 1)