def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    token_bytes = [bytes([value]) for value in range(256)]
    sequences = [list(text.encode("utf-8")) for text in corpus]

    vocab = []
    merges = []

    while len(token_bytes) < vocab_size:
        pair_counts = {}
        for sequence in sequences:
            for position in range(len(sequence) - 1):
                pair = (sequence[position], sequence[position + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

        if not pair_counts:
            break

        best_pair = max(
            pair_counts,
            key=lambda pair: (
                pair_counts[pair],
                token_bytes[pair[0]],
                token_bytes[pair[1]],
            ),
        )

        left_id, right_id = best_pair
        new_id = len(token_bytes)
        new_bytes = token_bytes[left_id] + token_bytes[right_id]
        token_bytes.append(new_bytes)

        vocab.append([new_id, list(new_bytes)])
        merges.append([left_id, right_id, new_id])

        new_sequences = []
        for sequence in sequences:
            merged_sequence = []
            position = 0
            while position < len(sequence):
                has_next = position < len(sequence) - 1
                if has_next and sequence[position] == left_id and sequence[position + 1] == right_id:
                    merged_sequence.append(new_id)
                    position += 2
                else:
                    merged_sequence.append(sequence[position])
                    position += 1
            new_sequences.append(merged_sequence)
        sequences = new_sequences

    return {"vocab": vocab, "merges": merges}