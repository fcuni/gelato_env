import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Union, Callable, List

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
    last_action: Optional[Dict[str, float]] = None
    local_reward: Optional[Dict[str, float]] = None
    global_reward: float = 0.0
    step: int = 0
    is_terminal: bool = False
    restock_period: int = 7

    @property
    def n_products(self):
        return len(self.products)

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
                                                   torch.tensor(product.stock),#torch.tensor(product.stock / self.max_stock),
                                                   torch.tensor(product.base_price),
                                                   torch.tensor(self.current_markdowns[product_id]),
                                                   torch.nn.functional.one_hot(torch.tensor(flavour_encoding),
                                                                               n_flavours),
                                                   ]).float())
        return torch.vstack(public_obs_tensor)
    



def default_init_state() -> GelateriaState:
    products = [Gelato(flavour=Flavour.VANILLA, base_price=1.0, stock=100, id=uuid.uuid4()),
                Gelato(flavour=Flavour.CHOCOLATE, base_price=1.0, stock=100, id=uuid.uuid4()),
                Gelato(flavour=Flavour.STRAWBERRY, base_price=1.0, stock=100, id=uuid.uuid4()),
                ]

    return GelateriaState(
        products={product.id: product for product in products},
        current_markdowns={product.id: 0.0 for product in products},
        last_markdowns={product.id: None for product in products},
        last_action={product.id: [] for product in products},
        local_reward={product.id: None for product in products},
    )


def init_state_from(products: List[Gelato]) -> GelateriaState:
    return GelateriaState(
        products={product.id: product for product in products},
        current_markdowns={product.id: 0.0 for product in products},
        last_markdowns={product.id: None for product in products},
        last_action={product.id: [] for product in products},
        local_reward={product.id: None for product in products},
    )
