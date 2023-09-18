from typing import Union, Dict, Sequence, Any, List, Tuple

import gym
import numpy as np
import torch

from env.gelateria import GelateriaState
from env.gelateria_env import GelateriaEnv
from utils.types import TensorType


class DefaultGelatoEnvWrapper(gym.Wrapper):
    def __init__(self, env: GelateriaEnv, action_output_dtype: type = int, output_as_tensor: bool = False,
                 info_output_as_array: bool = False):
        """
        Wrapper for the Gelati environment. This wrapper is used to transform the outputs of the environment to a format
        that are easier to pass to the agents.

        Args:
            env: GelateriaEnv to wrap.
            action_output_dtype: type of the action output. Either `int` or `float`.
            output_as_tensor: whether to output the state as a tensor or numpy array. If True, the state will be output
                as a tensor. If False, the state will be output as a numpy array.
            info_output_as_array: whether to output the info as a list or dictionary. If True, the info will be output
                as a list. If False, the info will be output as a dictionary.
        """

        super(DefaultGelatoEnvWrapper, self).__init__(env)
        self._info_output_as_array = info_output_as_array
        assert action_output_dtype in [int, float], "action_output_dtype must be either int or float"
        self._action_output_dtype = action_output_dtype
        self._output_as_tensor = output_as_tensor

    def step(self, action: Union[List[int], List[float], TensorType]) -> \
            Tuple[TensorType, TensorType, bool, Union[Dict[str, Any]]]:
        # Step the environment
        state, reward, done, info = self.env.step(self._transform_action(action),
                                                  action_dtype=self._action_output_dtype.__name__)
        # Apply transformation to the state observation
        transformed_state = self._transform_state(state)
        # Apply transformation to the reward
        transform_reward = self._transform_reward(reward, backward=False)
        # Apply transformation to the info
        transformed_info = self._transform_info(info, self.env.state)

        return transformed_state, transform_reward, done, transformed_info

    def _transform_state(self, state, target_obs_type: str = "public_obs") -> TensorType:
        """
           Extract the public observation (unless specified otherwise) from the state and convert it to a numpy array.

           Args:
               state: state to extract the observation from.
                target_obs_type: key name of the observation to extract from the state.

           Returns:
               numpy/tensor array of the public observation of the state.
           """
        if state is None:
            return None
        if target_obs_type in list(state.keys()):
            target_state = state[target_obs_type]

            if isinstance(target_state, torch.Tensor):
                if self._output_as_tensor:
                    return target_state
                return target_state.numpy()
            elif not isinstance(target_state, np.ndarray):
                numpy_arr = np.array(target_state)
                if self._output_as_tensor:
                    return torch.from_numpy(numpy_arr)
                return numpy_arr

    def _transform_action(self, action: Union[List[int], List[float], np.ndarray, torch.Tensor]) -> \
            Union[List[int], List[float]]:
        """
        Convert the action vector to a list of integers.
        """

        # convert the action vector to a numpy array
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        elif isinstance(action, list):
            action = np.array(action)

        # check if the action is in form of integers or floats
        if (action == action.astype(int)).all():
            transformed_action = action.tolist()
        else:
            transformed_action = (action * 100).astype(int).tolist()
        if self._action_output_dtype == int:
            return transformed_action
        return [float(a) / 100 for a in transformed_action]

    def _transform_reward(self, reward: Union[Sequence[float], Dict[str, float]], backward: bool = False) \
            -> Union[TensorType, Dict[str, float]]:
        if not backward:
            assert isinstance(reward, dict), f"Type mismatch when transforming reward from {type(reward)}. " \
                                             f"Forward mode: Reward must be a `Dict[str, float]` object."
            reward_numpy = np.array([reward[product_id] for product_id in self.env.state.products])
            if self._output_as_tensor:
                return torch.from_numpy(reward_numpy)
            return reward_numpy

        assert isinstance(reward, list), f"Type mismatch when transforming reward from {type(reward)}. " \
                                         f"Backward mode: Reward must be a `List[float]` object."
        return {product_id: reward[i] for i, product_id in enumerate(self.env.state.products)}

    def _transform_info(self, info: Dict[str, Any], env_state: GelateriaState) -> Dict[str, Any]:
        info = {"products": [product_id for product_id in env_state.products],
                "flavours": {product_id: env_state.products[product_id].flavour.value for product_id in
                             env_state.products},
                "stocks": {product_id: env_state.product_stocks[i] for i, product_id in enumerate(env_state.products)},
                "empty_shelf": {product_id: env_state.per_product_done_signal[i] for i, product_id in
                                enumerate(env_state.products)},
                "current_markdowns": {product_id: env_state.current_markdowns[product_id] for product_id in
                                      env_state.products},
                "current_price": {product_id: env_state.products[product_id].current_price(
                    markdown=env_state.current_markdowns[product_id]) for product_id in env_state.products},
                "step": env_state.step,
                "current_date": env_state.current_date,
                **info}
        if self._info_output_as_array:
            for k, v in info.items():
                if isinstance(v, dict):
                    info[k] = np.array(list(v.values()))
                    if self._output_as_tensor:
                        info[k] = torch.from_numpy(info[k])
                else:
                    info[k] = v
        return info

    def reset(self, get_info: bool = False) -> Union[Tuple[TensorType, Dict[str, Any]], TensorType]:
        """
        Reset the environment and return the initial state.

        Args:
            get_info: whether to return the info dictionary or not. Defaults to False.

        Returns:
            - the initial state of the environment.
            - the info dictionary if `get_info` is `True`.
        """

        state, reward, done, info = super().reset()

        # Apply transformation to the state observation
        transformed_state = self._transform_state(state)

        if get_info:
            # Apply transformation to the info
            transformed_info = self._transform_info(info, self.env.state)
            return transformed_state, transformed_info
        else:
            return transformed_state
