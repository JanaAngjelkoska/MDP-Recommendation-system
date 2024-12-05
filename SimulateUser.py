from random import random, randrange


def simulate(good_map=None, bad_map=None):
    """
    This function simulates random user actions while listening to ONE song, such that typical user actions are divided
    into 2 categories: positive and negative. The user will perform one or the other based on a single random decision.
    :return: Values for good and bad actions, possible rating and the name of the user
    """
    gave_rating = -1

    good_actions_map = {
        'repeat': False,
        'added_song': False,
        'shared': False,
    }
    bad_actions_map = {
        'skip': False,
        'removed_song': False
    }
    # random decision for whether the listener performs positive or negative actions

    if good_map is not None:
        good_actions_map = good_map
    if bad_map is not None:
        bad_actions_map = bad_map

    listening_percentage = random()
    if listening_percentage < 0.05:
        bad_actions_map['skip'] = True

    if listening_percentage > 0.4:
        for action in good_actions_map.keys():
            if random() > 0.8:
                good_actions_map[action] = True
        if random() > 0.8:
            gave_rating = randrange(3, 5)
    else:
        if random() > 0.5:
            bad_actions_map['removed_song'] = True
        if random() > 0.8:
            gave_rating = randrange(0, 2)

    return good_actions_map, bad_actions_map, listening_percentage, gave_rating