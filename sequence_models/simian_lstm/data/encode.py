import random


def add_encoding(encoding, call):
    max_existing = max(encoding.values())
    encoding[call] = max_existing + 1


def canonize_encoding(encoding):
    token_map = {token.lower(): token for token in encoding.keys()}
    tokens = sorted(list(token_map.keys()))
    rand = random.Random(1234)
    rand.shuffle(tokens)

    return {token_map[token]: i for i, token in enumerate(tokens, start=1)}