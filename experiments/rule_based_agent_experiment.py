from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.rule_based_agent.rule_based_agent import RuleBasedAgent
from utils.config import RuleBasedAgentExperimentConfig


def get_experiment_config():
    config = RuleBasedAgentExperimentConfig()
    return config


class RuleBasedAgentExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="RuleBasedAgentExperiment", config=get_experiment_config())
        self._config: RuleBasedAgentExperimentConfig = get_experiment_config()
        self._env = self.build_env_v2(self._config.env_config)

    def get_rl_model(self) -> RLAgent:
        rl_model = RuleBasedAgent(env=self._env, config=self._config.rule_based_agent_config)
        return rl_model


if __name__ == "__main__":
    RuleBasedAgentExperiment().run()





