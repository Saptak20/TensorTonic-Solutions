import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    """
    Returns: V (NumPy array of shape (n_states,))
    """
    
    returns = [[] for _ in range(n_states)]

    for episode in episodes:
        states = [s for s, r in episode]
        rewards = [r for s, r in episode]

        G = 0.0
        returns_from_t = [0.0] * len(episode)

        # Compute returns backwards
        for t in range(len(episode) - 1, -1, -1):
            G = rewards[t] + gamma * G
            returns_from_t[t] = G

        visited = set()

        # First-visit MC update
        for t, state in enumerate(states):
            if state not in visited:
                returns[state].append(returns_from_t[t])
                visited.add(state)

    V = np.zeros(n_states)

    for s in range(n_states):
        if returns[s]:
            V[s] = np.mean(returns[s])

    return V