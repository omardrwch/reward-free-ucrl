from agents.base_agent import BaseAgent


class RandomBaseline(BaseAgent):
    """
    Baseline for Reward-Free UCRL

    Random exploration policy
    """
    def run(self, total_samples):
        self.reset()
        sample_count = 0
        while sample_count < total_samples:
            state = self.env.reset()
            for _ in range(self.H):
                sample_count += 1
                action = self.env.action_space.sample()
                state = self.step(state, action)
        return self.estimation_error()
