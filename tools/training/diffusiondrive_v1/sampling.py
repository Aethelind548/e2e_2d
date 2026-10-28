"""Checkpoint committed samples, keeping worker prefetch outside resume state."""
from collections import deque

import torch
from torch.utils.data import Sampler


class CommittedSampler(Sampler):
    schema = 'b2d-committed-sampler-v1'

    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = []
        self.cursor = 0
        self.pending = deque()
        self.last_indices = []

    def __len__(self):
        return self.size

    def __iter__(self):
        # Dispatch uses a private cursor. Checkpoints record only acknowledged batches.
        # Workers may finish later items early; they do not advance committed state.
        self.pending.clear()
        generator = torch.Generator()
        generator.set_state(self.generator.get_state())
        order, cursor = list(self.order), self.cursor
        for _ in range(self.size):
            if cursor >= len(order):
                order = torch.randperm(self.size, generator=generator).tolist()
                cursor = 0
            index = order[cursor]
            cursor += 1
            self.pending.append((index, order, cursor, generator.get_state()))
            yield index

    def acknowledge(self, count):
        if count <= 0 or count > len(self.pending):
            raise RuntimeError('training batch does not match dispatched data cursor')
        self.last_indices = []
        for _ in range(count):
            index, order, cursor, rng = self.pending.popleft()
            self.last_indices.append(index)
            self.order = list(order)
            self.cursor = cursor
            self.generator.set_state(rng)
        return self.last_indices

    def state_dict(self):
        return {'schema': self.schema, 'size': self.size, 'order': self.order,
                'cursor': self.cursor, 'generator': self.generator.get_state()}

    def load_state_dict(self, state):
        if (state['schema'] != self.schema or state['size'] != self.size
                or sorted(state['order']) != list(range(self.size)) or not 0 <= state['cursor'] <= self.size):
            raise ValueError('invalid committed checkpoint data cursor')
        self.order = list(state['order'])
        self.cursor = state['cursor']
        self.generator.set_state(state['generator'])
        self.pending.clear()
