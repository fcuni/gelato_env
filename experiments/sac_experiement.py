# from env.reward.simple_reward import SimpleReward
# from env.reward.sales_uplift_reward import SalesUpliftReward
from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
# from models.mlp_sales import MLPLogSalesModel
from models.sac.sac_discrete import SACDiscrete
from utils.config import SACExperimentConfig

# from env.mask.simple_masks import BooleanMonotonicMarkdownsMask


def get_experiment_config():
    config = SACExperimentConfig()
    return config

class SACExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="SACExperiment_NewEnv", config=get_experiment_config())
        self._config: SACExperimentConfig = get_experiment_config()
        self._env = self.build_env_v2(self._config.env_config)

    def get_rl_model(self) -> RLAgent:
        rl_model = SACDiscrete(env=self._env, config=self._config.sac_config)
        return rl_model


if __name__ == "__main__":
    SACExperiment().run()

