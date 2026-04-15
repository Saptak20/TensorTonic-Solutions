def jaccard_similarity(set_a, set_b):
    # Convert lists to sets to remove duplicates
    A = set(set_a)
    B = set(set_b)
    
    # Handle edge case: both empty
    if not A and not B:
        return 0.0
    
    intersection = A & B
    union = A | B
    
    return len(intersection) / len(union)