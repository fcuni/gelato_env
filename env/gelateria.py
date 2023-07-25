import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Union, Callable, List

import numpy as np
import torch

from utils.enums import Flavour


@dataclass
class Gelato:
    flavour: Flavour
    base_price: float
    stock: int
    id: uuid.UUID = uuid.uuid4()

    def current_price(self, markdown: float) -> float:
        return self.base_price * (1 - markdown)

    def __repr__(self):
        return f"Gelato({self.flavour.value.upper()})"


@dataclass
class GelateriaState:
    products: Dict[str, Gelato]
    day_number: int = 0
    current_markdowns: Optional[Dict[str, float]] = None
    last_markdowns: Optional[Dict[str, float]] = None
    last_actions: Optional[Dict[str, List[float]]] = None


    historical_action_count: Optional[Dict[str, Dict[float, int]]] = None
    local_reward: Optional[Dict[str, float]] = None
    global_reward: float = 0.0
    step: int = 0
    is_terminal: bool = False
    # TODO: set the restock period back to 7 days
    restock_period: int = 1  # original: 7

    # TODO:to remove after testing
    sales_days: int = 365

    @property
    def n_products(self):
        return len(self.products)

    @property
    def product_stocks(self) -> List[int]:
        """Return the stock levels of the products in the environment."""
        return [product.stock for product in self.products.values()]

    @property
    def per_product_done_signal(self) -> np.ndarray:
        """Return the done signal for each product in the environment.
        The done signal is set to True if the product is out of stock."""
        return np.array([product.stock == 0 for product in self.products.values()])

    def historical_actions_count(self, product_id: Optional[str] = None) -> Dict[str, int]:
        """Return the number of times a product has been marked down in the past.
        If product_id is None, return the number of times of each product that has been marked down in the past.

        Args:
            product_id: (Optional) The id of the product to return the number of markdowns for.

        Returns:
            A dictionary mapping product ids to the number of times the product(s) being marked down in the past.
        """
        if product_id is None:
            return {pid: len(self.last_actions[pid]) for pid in self.products}

        assert product_id in self.products, f"Product {product_id} does not exist in the environment."
        return {product_id: len(self.last_actions[product_id])}

    def __post_init__(self):
        self.max_stock = max([product.stock for product in self.products.values()])

    def restock(self, restock_fct: Union[Callable[[Gelato], int], Dict[str, int]]):
        for product_id, stock in restock_fct.items():
            if isinstance(restock_fct, dict):
                self.products[product_id].stock = stock
            else:
                self.products[product_id].stock = restock_fct(self.products[product_id])

    def get_public_observations(self) -> torch.Tensor:
        flavour_one_hot = Flavour.get_flavour_encoding()
        n_flavours = len(Flavour.get_all_flavours())
        public_obs_tensor = []
        for product_id, product in self.products.items():
            flavour_encoding = flavour_one_hot[product.flavour.value]
            public_obs_tensor.append(torch.hstack([torch.tensor(self.day_number / 365),
                                                   torch.tensor(product.stock / self.max_stock),
                                                   torch.tensor(product.base_price),
                                                   torch.tensor(self.current_markdowns[product_id]),
                                                   torch.nn.functional.one_hot(torch.tensor(flavour_encoding),
                                                                               n_flavours),
                                                   ]).float())
        return torch.vstack(public_obs_tensor)

    def get_product_labels(self):
        """
        Construct and return the product labels from the state.

        Returns:
            list of product labels. (format: `{product name}_{product id}`).
            For example, `Gelato(VANILLA)_59a9d160-fc7c-4905-a7bd-5bd5a6ee293c`.
        """

        return [f"{str(self.products[product_id])}_{product_id}" for product_id in self.products.keys()]


def default_init_state() -> GelateriaState:
    products = [Gelato(flavour=Flavour.VANILLA, base_price=1.0, stock=100, id=uuid.uuid4()),
                Gelato(flavour=Flavour.CHOCOLATE, base_price=1.0, stock=100, id=uuid.uuid4()),
                Gelato(flavour=Flavour.STRAWBERRY, base_price=1.0, stock=100, id=uuid.uuid4()),
                ]

    return GelateriaState(
        products={product.id: product for product in products},
        current_markdowns={product.id: 0.0 for product in products},
        last_markdowns={product.id: None for product in products},
        last_actions={product.id: [] for product in products},
        local_reward={product.id: None for product in products},
    )


def init_state_from(products: List[Gelato]) -> GelateriaState:
    return GelateriaState(
        products={product.id: product for product in products},
        current_markdowns={product.id: 0.0 for product in products},
        last_markdowns={product.id: None for product in products},
        last_actions={product.id: [] for product in products},
        local_reward={product.id: None for product in products},
        historical_action_count={product.id: {} for product in products}
    )
