# import libs
from collections import namedtuple

# a named tuple representing a single transition in our environment
Transition = namedtuple('Transition',
                       		('state', 'action', 'next_state', 'reward')
                       )


class ReplayMemory(object):
    """
	    A cyclic buffer of bounded size that holds the transitions observed recently. It also implements a .sample() 
	    method for selecting a random batch of transitions for training.

	    ...

	    Attributes
	    ----------
	    capacity : float
	        Model capacity for the memory.
	    memory : list
	        Last occurrences.
	    position : float
	        Position in memory to store.

	    Methods
	    -------
	    push()
	        Saves a transition.
	    sample(batch_size)
	        Select a random batch of transitions for training.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
    	""" Select a random batch of transitions for training. """
        return random.sample(self.memory, batch_size)


    def __len__(self):
    	""" Return memory size. """
        return len(self.memory)
