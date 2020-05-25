"""
Baseline for Reward-Free UCRL

Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 

Sample n transitions from each state-action pair to estimate the model \hat{P},
then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
"""
import numpy as np
from agents.base_agent import BaseAgent


class MB_QVI(BaseAgent):
    r"""
    Model-based Q-value iteration (Azar et al., 2012), adapted to the reward free case. 
    
    Sample n transitions from each state-action pair to estimate the model \hat{P},
    then run value iteration with (true_reward, \hat{P}) to estimate the optimal Q-function
    """
    def run(self, total_samples):
        self.reset()
        n_samples_per_pair = int(np.ceil(total_samples/(self.S*self.A)))
        for ss in range(self.S):
            for aa in range(self.A):
                for _ in range(n_samples_per_pair):
                    # sample transition from (ss, aa)
                    self.env.reset(ss)   # put env in state ss     <---------  NOT IN GYM, ATTENTION HERE
                    self.step(ss, aa)
        return self.estimation_error()