"""
Baseline for Reward-Free UCRL

Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 

Sample n transitions from each state-action pair to estimate the model \hat{P},
then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
"""
import random
from itertools import product

import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent


class MB_QVI(BaseAgent):
    r"""
    Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 
    
    Sample n transitions from each state-action pair to estimate the model \hat{P},
    then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
    """
    name: str = "Generative Model"

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        # Split budget n into pSA uniform + q random transitions
        uniform_transitions = list(product(range(self.S), range(self.A), range(total_samples // (self.S*self.A))))
        supplementary_transitions = list(product(range(self.S), range(self.A), [1]))
        random.shuffle(supplementary_transitions)
        supplementary_transitions = supplementary_transitions[:(total_samples % (self.S*self.A))]
        assert total_samples == len(uniform_transitions) + len(supplementary_transitions)
        for ss, aa, _ in uniform_transitions + supplementary_transitions:
            # sample transition from (ss, aa)
            self.env.reset(ss)   # put env in state ss     <---------  NOT IN GYM, ATTENTION HERE
            self.step(ss, aa)
        return self.result_dataframe(total_samples)
