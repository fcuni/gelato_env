from env.reward.simple_reward import SimpleReward
from env.reward.sales_uplift_reward import SalesUpliftReward
from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.mlp_sales import MLPLogSalesModel
from models.sac.sac_discrete import SACDiscrete
from utils.config import SACExperimentConfig

from env.mask.simple_masks import BooleanMonotonicMarkdownsMask


def get_experiment_config():
    config = SACExperimentConfig()
    return config

class SACExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="SACExperiment_NewEnv", config=get_experiment_config())
        self._config: SACExperimentConfig = get_experiment_config()
        #
        # supervised_model = self.get_supervised_model(MLPLogSalesModel)

        # self._env = self.build_env(
        #     init_state=self._config.data_generation_config.init_state,
        #     supervised_model=MLPLogSalesModel,
        #     reward=SimpleReward(waste_penalty=0.0),
        #     # reward=SalesUpliftReward(sales_model=self.get_supervised_model(MLPLogSalesModel), markdown_penalty=1.0),
        #     action_mask_fn=BooleanMonotonicMarkdownsMask,
        #     restock_fct=None
        # )
        self._env = self.build_env_v2()

    def get_rl_model(self) -> RLAgent:
        rl_model = SACDiscrete(env=self._env, config=self._config.sac_config)
        return rl_model


if __name__ == "__main__":
    SACExperiment().run()

