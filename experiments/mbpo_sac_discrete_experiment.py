from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.mbpo.mbpo import MBPO
from models.sac.sac_discrete import SACDiscrete
from utils.config import MBPOExperimentConfig


def get_experiment_config():
    config = MBPOExperimentConfig()
    return config


class MbpoSacDiscreteExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="MbpoSacDiscreteExperiment", config=get_experiment_config())
        self._config: MBPOExperimentConfig = get_experiment_config()
        self._env = self.build_env_v2(self._config.env_config)

    def get_agent(self) -> RLAgent:
        agent = SACDiscrete(env=self._env, config=self._config.sac_config)
        return agent

    def get_rl_model(self) -> RLAgent:
        sac_agent = self.get_agent()
        rl_model = MBPO(env=self._env, agent=sac_agent, config=self._config.mbpo_config)
        return rl_model


if __name__ == "__main__":
    MbpoSacDiscreteExperiment().run()
