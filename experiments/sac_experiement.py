from data_generators.data_generators import DataGenerator
from env.gelateria_env import GelateriaEnv
from env.reward.sales_uplift_reward import SalesUpliftReward
from env.reward.simple_reward import SimpleReward
from env.reward.simple_reward_with_early_term_penalty import SimpleRewardEarlyTerminationPenalty
from models.mlp_sales import MLPLogSalesModel
# from models.simple_sales import SimpleLinearSalesModel

from models.sac.agent_old import SACAgent
from models.sac.sac_discrete import SACDiscrete
from models.simple_sales import SimpleLinearSalesModel
from utils.config import SACExperimentConfig

from env.mask.simple_masks import BooleanMonotonicMarkdownsMask, NoRestrictionBooleanMask


def get_experiment_config():
    config = SACExperimentConfig()
    return config


class SACExperiment:
    def __init__(self):
        self._config: SACExperimentConfig = get_experiment_config()

    def _get_dataset_generator(self):
        return DataGenerator(
            config=self._config.data_generation_config, dataloader_config=self._config.dataloader_config
        )

    def get_supervised_model(self):
        dummy_dataloader, _ = self._get_dataset_generator().get_train_val_dataloaders()
        input_dim = next(iter(dummy_dataloader))["public_obs"].shape[-1]
        model = MLPLogSalesModel(input_dim=input_dim, name="mlp_sales", config=self._config.net_config)
        # model = SimpleLinearSalesModel(input_dim=input_dim, name="simple_sales", config=self._config.net_config)
        model.load()
        return model

    def get_rl_model(self):
        rl_model = SACDiscrete(env=self.build_env(), config=self._config)
        # rl_model = SACAgent(env=self.build_env(), config=self._config)
        return rl_model


    def build_env(self):
        # TODO: set waste penalty back to 0.0 after testing
        supervised_model = self.get_supervised_model()
        reward = SalesUpliftReward(sales_model=supervised_model,
                                   markdown_penalty=self._config.sac_config.markdown_penalty,
                                   waste_penalty=self._config.sac_config.waste_penalty)
        # reward = SimpleReward(waste_penalty=0.0)
        # reward = SimpleRewardEarlyTerminationPenalty(waste_penalty=0.0, early_term_penalty=0.0)
        init_state = self._config.data_generation_config.init_state
        restock_fct = None
        env = GelateriaEnv(init_state=init_state,
                           sales_model=supervised_model,
                           reward=reward,
                           mask_fn=self._config.sac_config.mask_fn,
                           restock_fct=restock_fct)

        return env

    def run(self):
        agent = self.get_rl_model()
        agent.train()

        # if self._config.optimiser_config.path_to_model is not None:
        #     agent.save()


if __name__ == "__main__":
    SACExperiment().run()

