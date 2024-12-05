class MDPSolver:
    def __init__(self, states, gamma=0.9, threshold=0.001):
        """
        :param states: A dictionary of states, where keys are state names and values are State instances
        :param gamma: Discount factor for future rewards
        :param threshold: Convergence threshold for value iteration
        """
        self.states = states
        self.gamma = gamma
        self.threshold = threshold
        self.state_values = {state.name: 0 for state in states.values()}
        self.iter = 0
        self.epsilon = 10

    def value_iteration(self):
        """
        Perform value iteration to find the optimal state values
        """
        self.iter += 1
        if self.iter > 400:
            self.epsilon = self.epsilon / 10

        while True:
            delta = 0
            for state in self.states.values():

                # allow exploration with epsilon greedy
                if np.random.random() < self.epsilon:
                    return state.neighbours[randrange(0, len(state.neighbours) - 1)]
                max_value = float('-inf')
                prob = 0
                for neighbor in state.neighbours:
                    transition_prob = state.transition_probability_to(neighbor)

                    reward = state.reward
                    # prob += transition_prob # for testing purposes
                    next_value = reward + self.gamma * self.state_values[neighbor.name]
                    max_value = max(max_value, transition_prob * next_value)
                    print("Max value for state ", state.name, " is ", max_value)
                # print("PROBABILITY CHECK --------------------------", prob)

                delta = max(delta, abs(max_value - self.state_values[state.name]))
                self.state_values[state.name] = max_value

            if delta < self.threshold:
                break

        return self.state_values

    def get_optimal_policy(self):
        """
        Derive the optimal policy based on the computed state values
        """
        policy = {}
        for state in self.states.values():
            best_action = None
            best_value = float('-inf')
            prob = 0
            for neighbor in state.neighbours:
                transition_prob = state.transition_probability_to(neighbor)
                prob += transition_prob
                expected_value = transition_prob * (state.reward + self.gamma * self.state_values[neighbor.name])
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = neighbor

            policy[state.name] = best_action.name if best_action else None

        return policy
