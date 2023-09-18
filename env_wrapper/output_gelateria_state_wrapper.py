from copy import deepcopy
from typing import Union, List, Dict, Any, Tuple

import gym

from env.gelateria import GelateriaState
from env.gelateria_env import GelateriaEnv
from utils.types import TensorType


class OutputGelateriaStateWrapper(gym.Wrapper):
    def __init__(self, env: Union[GelateriaEnv, gym.Wrapper]):
        super(OutputGelateriaStateWrapper, self).__init__(env)

    def step(self, action: Union[List[int], List[float], TensorType]) -> \
            Tuple[GelateriaState, TensorType, bool, Union[Dict[str, Any]]]:
        # Step the environment
        state, reward, done, info = self.env.step(action)
        # get the gelateria state as observation
        gelateria_state = deepcopy(self.env.state)

        return gelateria_state, reward, done, info

    def reset(self, get_info:bool = False) -> Union[GelateriaState, Tuple[GelateriaState, Dict[str, Any]]]:
        # reset the environment
        if get_info:
            # assert that the environment has a reset method take a get_info argument
            import inspect
            assert "get_info" in inspect.signature(self.env.reset).parameters, \
                "The reset method in this env does not have a get_info argument"

            _, info = self.env.reset(get_info=get_info)

        else:
            self.env.reset()

        gelateria_state: GelateriaState = deepcopy(self.env.state)

        if get_info:
            return gelateria_state, info
        return gelateria_state

