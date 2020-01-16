import random
from collections import namedtuple

# Create a named tuple to more semantically transform transitions and batches of transitions
Transition = namedtuple('transition', ('state', 'action', 'reward', 'state_', 'done', 'raw_state'))

# Memory which allows for storing and sampling batches of transtions
class ReplayBuffer(object):
    def __init__(self, size=1e6):
        self.buffer = []
        self.max_size = size
        self.pointer = 0

    # Adds a single transitions to the memory buffer
    def add_transition(self, *args):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        
        self.buffer[self.pointer] = Transition(*args)
        self.pointer = int((self.pointer + 1) % self.max_size)

    # Samples a batch of transitions
    def sample_batch(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)

        # Converts batch of transitions to transitions of batches
        batch = Transition(*zip(*batch))

        return batch

    def __len__(self):
        return len(self.buffer)