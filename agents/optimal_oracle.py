from agents.base_agent import BaseAgent


class Optimal(BaseAgent):
    """
    Optimal policy
    """
    def run(self, total_samples):
        self.reset()
        sample_count = 0
        while sample_count < total_samples:
            state = self.env.reset()
            for h in range(self.H):
                sample_count += 1
                action = self.trueQ[h, state].argmax()
                state = self.step(state, action)
        return self.estimation_error()
