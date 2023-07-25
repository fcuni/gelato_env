from env.reward.simple_reward import SimpleReward
from experiments.experiment import BaseExperiment
from models.base_rl_agent import RLAgent
from models.mlp_sales import MLPLogSalesModel
from models.td_zero import TDZero
from utils.config import TDZeroExperimentConfig

from env.mask.simple_masks import MonotonicMarkdownsMask

def get_experiment_config():
    config = TDZeroExperimentConfig()
    return config


class TDZeroExperiment(BaseExperiment):
    def __init__(self):
        super().__init__(name="TDZeroExperiment", config=get_experiment_config())
        self._config: TDZeroExperimentConfig = get_experiment_config()
        self._env = self.build_env(
            init_state=self._config.data_generation_config.init_state,
            supervised_model=MLPLogSalesModel,
            reward=SimpleReward(waste_penalty=0.0),
            action_mask_fn=MonotonicMarkdownsMask,
            restock_fct=None
        )

    def get_rl_model(self) -> RLAgent:
        rl_model = TDZero(env=self._env, config=self._config.td_zero_config)
        return rl_model


if __name__ == "__main__":
    TDZeroExperiment().run()
