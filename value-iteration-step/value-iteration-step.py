def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    new_values = []

    for state in range(len(values)):
        best_action_value = float("-inf")

        for action in range(len(transitions[state])):
            expected_next_value = 0.0
            for next_state, probability in enumerate(transitions[state][action]):
                expected_next_value += probability * values[next_state]

            action_value = rewards[state][action] + gamma * expected_next_value

            if action_value > best_action_value:
                best_action_value = action_value

        new_values.append(float(best_action_value))

    return new_values