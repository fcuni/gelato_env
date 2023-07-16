from enum import Enum
from typing import List

import numpy as np


class Flavour(Enum):
    VANILLA = "vanilla"
    CHOCOLATE = "chocolate"
    STRAWBERRY = "strawberry"
    MINT = "mint"
    COOKIES_AND_CREAM = "cookies_and_cream"
    COFFEE = "coffee"
    PISTACHIO = "pistachio"
    LEMON = "lemon"
    MANGO = "mango"
    RASPBERRY = "raspberry"

    @classmethod
    def get_all_flavours(cls):
        return [flavour.value for flavour in cls]

    @classmethod
    def get_flavour_encoding(cls):
        one_hot = {}
        for idx, flavour in zip(range(len(cls)), cls):
            one_hot[flavour.value] = idx
        return one_hot

    @classmethod
    def get_flavour_from_one_hot_encoding(cls, one_hot: np.ndarray) -> List["Flavour"]:
        """Returns the flavour corresponding to the one-hot encoding."""
        one_hot = np.atleast_2d(one_hot)
        assert one_hot.shape[1] == len(
            cls.get_all_flavours()), "One-hot encoding must have the same length as the number of flavours."
        flavour_ids = one_hot.argmax(keepdims=True, axis=1)
        return [cls(x) for x in np.array(cls.get_all_flavours())[flavour_ids]]
