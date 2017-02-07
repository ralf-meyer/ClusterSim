# Markov Chain Monte Carlo Methods

import numpy as _np

class MetropolisHastings(object):
    def __init__(self, cluster, method):
        self.cluster = cluster
        self.markov_chain = []

    def step(self):
        

        # Generate a trial geometry
	# Choose random atom:
        ind1 = _np.random.randint(0, len(self.cluster.atoms))


        # Calculate acceptance ratio

        # Append the result to the Markov Chain
        
