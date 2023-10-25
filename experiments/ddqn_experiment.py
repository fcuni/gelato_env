from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.ddqn.ddqn import DDQN
from utils.config import DDQNExperimentConfig


def get_experiment_config():
    config = DDQNExperimentConfig()
    return config


class DDQNExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="DdqnExperiment", config=get_experiment_config())
        self._config: DDQNExperimentConfig = get_experiment_config()
        self._env = self.build_env(self._config.env_config)

    def get_rl_model(self) -> RLAgent:
        rl_model = DDQN(env=self._env, config=self._config.dqn_config)
        return rl_model


if __name__ == "__main__":
    DDQNExperiment().run()
