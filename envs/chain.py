import numpy as np
from .finitemdp import FiniteMDP


class Chain(FiniteMDP):
    """
    Simple chain environment. 
    Reward 0.05 in initial state, reward 1.0 in final state.

    :param L:         length of the chain
    :param fail_prob: fail probability 
    """
    def __init__(self, L: int, fail_prob: float) -> None:
        assert L >= 2
        self.L = L
        states = list(range(L))
        action_sets = [[0, 1] for _ in states]

        # transition probabilities
        P = np.zeros((L, 2, L))
        for ss in range(L):
            for aa in range(2):
                if ss == 0:
                    P[ss, 0, ss]   = 1.0-fail_prob  # action 0 = don't move
                    P[ss, 1, ss+1] = 1.0-fail_prob  # action 1 = right
                    P[ss, 0, ss+1] = fail_prob  
                    P[ss, 1, ss]   = fail_prob          
                elif ss == L-1:
                    P[ss, 0, ss-1] = 1.0-fail_prob  # action 0 = left
                    P[ss, 1, ss]   = 1.0-fail_prob  # action 1 = don't move
                    P[ss, 0, ss]   = fail_prob  
                    P[ss, 1, ss-1] = fail_prob 
                else:
                    P[ss, 0, ss-1] = 1.0-fail_prob  # action 0 = left
                    P[ss, 1, ss+1] = 1.0-fail_prob  # action 1 = right
                    P[ss, 0, ss+1] = fail_prob  
                    P[ss, 1, ss-1] = fail_prob 
        
        # init base class
        super().__init__(states, action_sets, P)


        # mean reward
        S = self.observation_space.n 
        A = self.action_space.n
        self.mean_R  = np.zeros((S, A))
        for ss in range(S):
            for aa in range(A):
                mean_r = 0
                for ns in range(S):
                    mean_r += self.reward_fn(ss, aa, ns)*self.P[ss, aa, ns]
                self.mean_R[ss, aa] = mean_r

    def reward_fn(self, state: int, action: int, next_state: int) -> float:
        """
        Reward function
        """
        if state == self.L-1 and next_state == self.L-1:
            return 1.0
        if state == 0 and next_state == 0:
            return 0.05
        return 0
