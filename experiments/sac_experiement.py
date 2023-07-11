from data_generators.data_generators import DataGenerator
from env.gelateria_env import GelateriaEnv
from env.reward.simple_reward import SimpleReward
from models.mlp_sales import MLPLogSalesModel
from models.simple_sales import SimpleLinearSalesModel

# from models.td_zero import TDZero
from models.sac.agent import SACAgent
from utils.config import ExperimentConfig

from env.mask.simple_masks import BooleanMonotonicMarkdownsMask


def get_experiment_config():
    config = ExperimentConfig()
    return config


class SACExperiment:
    def __init__(self):
        self._config: ExperimentConfig = get_experiment_config()

    def _get_dataset_generator(self):
        return DataGenerator(
            config=self._config.data_generation_config, dataloader_config=self._config.dataloader_config
        )

    def get_supervised_model(self):
        dummy_dataloader, _ = self._get_dataset_generator().get_train_val_dataloaders()
        input_dim = next(iter(dummy_dataloader))["public_obs"].shape[-1]
        #model = MLPLogSalesModel(input_dim=input_dim, name="mlp_sales", config=self._config.net_config)
        model = SimpleLinearSalesModel(input_dim=input_dim, name="simple_sales", config=self._config.net_config)
        model.load()
        return model

    def get_rl_model(self):
        rl_model = SACAgent(env=self.build_env(), name="sac", config=self._config.optimiser_config)
        return rl_model


    def build_env(self):
        reward = SimpleReward(waste_penalty=0.0)
        init_state = self._config.data_generation_config.init_state
        restock_fct = None
        env = GelateriaEnv(init_state=init_state,
                           sales_model=self.get_supervised_model(),
                           reward=reward,
                           mask_fn=BooleanMonotonicMarkdownsMask,
                           restock_fct=restock_fct)

        return env

    def run(self):
        agent = self.get_rl_model()
        agent.train()

        # if self._config.optimiser_config.path_to_model is not None:
        #     agent.save()


if __name__ == "__main__":
    SACExperiment().run()

