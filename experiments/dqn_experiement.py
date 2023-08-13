from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.dqn.dqn import DQN
from utils.config import DQNExperimentConfig


def get_experiment_config():
    config = DQNExperimentConfig()
    return config


class DQNExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="DQNExperiment", config=get_experiment_config())
        self._config: DQNExperimentConfig = get_experiment_config()
        self._env = self.build_env_v2()

    def get_rl_model(self) -> RLAgent:
        rl_model = DQN(env=self._env, config=self._config.dqn_config)
        return rl_model


if __name__ == "__main__":
    DQNExperiment().run()
