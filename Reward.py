from SimulateUser import simulate


class Reward:
    def __init__(self, current_song, previous_songs):
        """
        Reward class for forming and calculating the reward AFTER the used left the state (switched songs)
        :param current_song: An instance of the currently playing song
        :param previous_songs: An instance of the last two songs played
        """
        self.current_song = current_song
        self.last_songs = previous_songs

    def calculate_reward(self, simulation=None):

        if simulation is not None:
            good, bad, listening_time, gave_rating = simulation
        else:
            good, bad, listening_time, gave_rating = simulate()
        # print(good, " - GOOD") # testing
        # print(bad, " - BAD")
        # print(listening_time, " - LISTENING TIME")
        # print(gave_rating, " - GAVE RATING")
        #  Negative reward for listening for a short time
        if listening_time < 0.025:
            listening_time = listening_time * (-4)
        elif 0.025 < listening_time < 0.25:
            listening_time = listening_time * (-2)

        positive = 0
        negative = 0
        for key in good.keys():
            if good[key]:
                positive += 1

        for key in bad.keys():
            if bad[key]:
                negative -= 1
        if simulation is None:
            if 0 <= gave_rating <= 2:
                negative -= 2
            if 4 <= gave_rating <= 5:
                positive += 1

            if positive == 0 and negative == 0:
                listening_time = listening_time / 2

        return (listening_time + positive + negative) / 4
