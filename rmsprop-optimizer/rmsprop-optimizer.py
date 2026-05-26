import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """

    # Convert inputs to numpy arrays
    w = np.array(w, dtype=float)
    g = np.array(g, dtype=float)
    s = np.array(s, dtype=float)

    # Update running squared gradient average
    s = beta * s + (1 - beta) * (g ** 2)

    # Update parameters
    w = w - (lr / (np.sqrt(s) + eps)) * g

    return w, s