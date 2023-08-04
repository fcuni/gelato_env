import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Union, Callable, List

import numpy as np
import torch

from utils.enums import Flavour


@dataclass
class Gelato:
    flavour: Flavour
    base_price: float
    stock: int
    restock_possible: bool = True
    id: uuid.UUID = uuid.uuid4()

    def current_price(self, markdown: float) -> float:
        return self.base_price * (1 - markdown)

    def __repr__(self):
        return f"Gelato({self.flavour.value.upper()})"


@dataclass
class GelateriaState:
    products: Dict[str, Gelato]
    day_number: int = 0
    current_date: Optional[datetime] = None
    current_markdowns: Optional[Dict[str, float]] = None
    last_markdowns: Optional[Dict[str, float]] = None
    last_actions: Optional[Dict[str, List[float]]] = None
    historical_sales: Optional[Dict[str, List[float]]] = None

    historical_action_count: Optional[Dict[str, Dict[float, int]]] = None
    local_reward: Optional[Dict[str, float]] = None
    global_reward: float = 0.0
    step: int = 0
    is_terminal: bool = False
    # TODO: set the restock period back to 7 days
    restock_period: int = 366  # original: 7


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
        for product_id, product in self.products.items():
            # only restock if the product can be restocked
            if product.restock_possible:
                if isinstance(restock_fct, dict):
                    self.products[product_id].stock = restock_fct[product_id]
                else:
                    self.products[product_id].stock = restock_fct(product)

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
            # TODO: currently the testing of the new public obs is moved to gelateria_env_v2
            # public_obs_tensor.append(torch.hstack([torch.tensor(self.step,
            #                                        torch.tensor(product.stock),
            #                                        torch.tensor(product.base_price),
            #                                        torch.tensor(self.current_markdowns[product_id]),
            #                                        torch.nn.functional.one_hot(torch.tensor(flavour_encoding),
            #                                                                    n_flavours),
            #                                        ]).float())
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
        historical_sales={product.id: [] for product in products}
    )


def init_state_from(products: List[Gelato]) -> GelateriaState:
    return GelateriaState(
        products={product.id: product for product in products},
        current_markdowns={product.id: 0.0 for product in products},
        last_markdowns={product.id: None for product in products},
        last_actions={product.id: [] for product in products},
        local_reward={product.id: None for product in products},
        historical_sales={product.id: [] for product in products},
        # historical_action_count={product.id: {} for product in products}
    )


# TODO: to move this out after finish
def default_init_state_new() -> GelateriaState:
    import pandas as pd

    # load the masked dataset
    df = pd.read_csv("masked_dataset.csv")
    # sort the dataset by date
    df['calendar_date'] = pd.to_datetime(df['calendar_date'])
    df.sort_values(by='calendar_date', inplace=True)
    # query the last date in the dataset
    last_date = df['calendar_date'].max()

    products = []
    current_markdowns = {}
    # Loop through the rows in the filtered DataFrame
    for index, row in df[df['calendar_date'] == last_date].iterrows():
        # Access the values of each column for the current row
        products_id = uuid.uuid4()
        flavour = Flavour(row['flavour'])
        base_price = float(row['full_price_masked'])
        stock = int(row['stock'])
        current_markdowns[products_id] = row['markdown']
        restock_possible = False
        products += [Gelato(flavour=flavour, base_price=base_price, stock=stock, id=products_id,
                            restock_possible=restock_possible)]

    return GelateriaState(
        products={product.id: product for product in products},
        current_markdowns=current_markdowns,
        last_markdowns={product.id: None for product in products},
        last_actions={product.id: [] for product in products},
        local_reward={product.id: None for product in products},
        historical_sales={product.id: [] for product in products},
        current_date=last_date
    )
