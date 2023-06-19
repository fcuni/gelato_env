import numpy as np

from env.gelateria import GelateriaState
from env.mask.action_mask import ActionMask


class MonotonicMarkdownsMask(ActionMask):

    def __init__(self):
        super().__init__(name="MonotonicMarkdownMask")

    def __call__(self, state: GelateriaState):
        mask = np.zeros((len(state.products), 101))
        for idx, markdown in enumerate(state.current_markdowns.values()):
            mask[idx, :int(markdown * 100)] = -np.inf
        return mask.squeeze()
