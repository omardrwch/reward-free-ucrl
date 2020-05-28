from agents.base_agent import BaseAgent
from utils import random_argmax


class Optimal(BaseAgent):
    name: str = "Optimal policy"

    def run(self, total_samples: int) -> float:
        self.reset()
        sample_count = 0
        while sample_count < total_samples:
            state = self.env.reset()
            for h in range(self.H):
                sample_count += 1
                action = random_argmax(self.trueQ[h, state])
                state = self.step(state, action)
        return self.estimation_error()
