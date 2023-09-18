from enum import Enum
from typing import List

import numpy as np


class Flavour(Enum):

    # #### OLD FLAVOURS ####
    # VANILLA = "vanilla"
    # CHOCOLATE = "chocolate"
    # STRAWBERRY = "strawberry"
    # MINT = "mint"
    # COOKIES_AND_CREAM = "cookies_and_cream"
    # COFFEE = "coffee"
    # PISTACHIO = "pistachio"
    # LEMON = "lemon"
    # MANGO = "mango"
    # RASPBERRY = "raspberry"

    # #### NEW FLAVOURS ####
    PISTACHIO = "Pistachio"
    SAFFRON_CARDAMOM = "Saffron Cardamom"
    ROSEWATER_ALMOND = "Rosewater Almond"
    LEMON_BASIL = "Lemon Basil"
    BLACKBERRY_SAGE = "Blackberry Sage"
    WHITE_CHOCOLATE_RASPBERRY = "White Chocolate Raspberry"
    MANGO_CHILI_LIME = "Mango Chili Lime"
    LAVENDER_HONEY = "Lavender Honey"
    EARL_GREY_TEA = "Earl Grey Tea"
    ROASTED_BANANA_CARAMEL = "Roasted Banana Caramel"
    FIG_BALSAMIC = "Fig Balsamic"
    COCONUT_LEMONGRASS = "Coconut Lemongrass"
    BLUEBERRY_CHEESECAKE = "Blueberry Cheesecake"
    HAZELNUT_PRALINE = "Hazelnut Praline"
    BLOOD_ORANGE_SORBETTO = "Blood Orange Sorbetto"
    CINNAMON_TOAST_CRUNCH = "Cinnamon Toast Crunch"
    PINEAPPLE_BASIL = "Pineapple Basil"
    MATCHA_GREEN_TEA = "Matcha Green Tea"
    DULCE_DE_LECHE = "Dulce de Leche"
    GUAVA_PASSIONFRUIT = "Guava Passionfruit"
    BUTTER_PECAN = "Butter Pecan"
    POMEGRANATE_ROSE = "Pomegranate Rose"
    CHERRY_AMARETTO = "Cherry Amaretto"
    DARK_CHOCOLATE_RASPBERRY_TRUFFLE = "Dark Chocolate Raspberry Truffle"
    SEA_SALT_CARAMEL = "Sea Salt Caramel"
    MAPLE_BACON = "Maple Bacon"
    COOKIES_AND_CREAM = "Cookies and Cream"
    WATERMELON_MINT = "Watermelon Mint"
    TIRAMISU = "Tiramisu"
    MINT_CHOCOLATE_CHIP = "Mint Chocolate Chip"
    GINGER_TURMERIC = "Ginger Turmeric"
    CHAI_TEA_LATTE = "Chai Tea Latte"
    GRAPEFRUIT_CAMPARI = "Grapefruit Campari"

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
