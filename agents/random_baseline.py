from agents.base_agent import BaseAgent
import numpy as np
import pandas as pd

class RandomBaseline(BaseAgent):
    """
    Baseline for Reward-Free UCRL

    Random exploration policy
    """
    name: str = "Random Policy"

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        sample_count = 0
        while sample_count < total_samples:
            state = self.env.reset()
            for _ in range(self.H):
                sample_count += 1
                action = np.random.randint(self.env.action_space.n)
                state = self.step(state, action)
        return self.result_dataframe(total_samples)
