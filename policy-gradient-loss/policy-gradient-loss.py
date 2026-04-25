def policy_gradient_loss(log_probs, rewards, gamma):
    T = len(rewards)
    
    # Step 1: Compute discounted returns
    returns = [0] * T
    G = 0
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
    
    # Step 2: Mean baseline
    mean_return = sum(returns) / T
    
    # Step 3: Compute loss
    loss = 0
    for log_p, G_t in zip(log_probs, returns):
        advantage = G_t - mean_return
        loss += log_p * advantage
    
    # Negative mean (gradient ascent → descent)
    loss = -loss / T
    
    return loss