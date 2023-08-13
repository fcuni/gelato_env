from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.dqn.dqn import DQN
from models.mbpo.mbpo import MBPO
from utils.config import MBPOExperimentConfig


def get_experiment_config():
    config = MBPOExperimentConfig()
    return config


class MbpoDqnExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="MbpoDqnExperiment", config=get_experiment_config())
        self._config: MBPOExperimentConfig = get_experiment_config()
        self._env = self.build_env_v2()

    def get_agent(self) -> RLAgent:
        agent = DQN(env=self._env, config=self._config.dqn_config)
        return agent

    def get_rl_model(self) -> RLAgent:
        dqn_agent = self.get_agent()
        rl_model = MBPO(env=self._env, agent=dqn_agent, config=self._config.mbpo_config)
        return rl_model


if __name__ == "__main__":
    MbpoDqnExperiment().run()
