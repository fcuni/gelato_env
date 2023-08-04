import abc
from typing import Sequence, List, Union, Dict, Any, Tuple, Optional, Callable

import torch

from models.sales.sales_uplift_models import SalesUpliftModel
from models.sales.base_sales_models import BaseSalesModel


class SalesPredictionModel:
    def __init__(self, base_sales_model: BaseSalesModel, uplift_model: SalesUpliftModel,
                 base_sales_input_transform_fn: Optional[Callable] = None):
        self._base_sales_model = base_sales_model
        self._uplift_model = uplift_model
        self._base_sales_input_transform_fn = base_sales_input_transform_fn

    @property
    def base_sales_model_info(self):
        return self._base_sales_model.info

    def _transform_inputs_from_gym(self, inputs: Sequence[Sequence[float]]) -> torch.Tensor:
        raise NotImplementedError

    def _get_base_sales(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._base_sales_model.get_sales(inputs, input_transform_fn=self._base_sales_input_transform_fn)

    def _get_uplift(self, markdown: Sequence[float]) -> List[float]:
        return self._uplift_model(markdown)

    @abc.abstractmethod
    def get_sales(self, inputs: torch.Tensor, output_info: bool = False, *args, **kwargs) -> \
            Union[List[float], Tuple[List[float], Dict[str, Any]]]:
        raise NotImplementedError


class GenericSalesPredictionModel(SalesPredictionModel):
    def __init__(self, base_sales_model: BaseSalesModel, uplift_model: SalesUpliftModel,
                 base_sales_input_transform_fn: Optional[Callable] = None):
        super().__init__(base_sales_model, uplift_model, base_sales_input_transform_fn)

    @property
    def base_sales_model_info(self):
        return self._base_sales_model.info

    def get_sales(self, inputs: torch.Tensor, output_info: bool = False) -> \
            Union[List[float], Tuple[List[float], Dict[str, Any]]]:

        markdowns = inputs[:, 0].detach().cpu().numpy().flatten()
        base_sales = self._get_base_sales(inputs).detach().cpu().flatten()
        sales_uplifts = torch.tensor(self._get_uplift(markdowns))
        sales = (base_sales * sales_uplifts).tolist()
        # log info
        info = {"base_sales": base_sales.tolist(), "sales_uplift": sales_uplifts.tolist()}

        if output_info:
            return sales, info
        else:
            return sales

#
# class SeparateFlavourSalesPredictionModel(SalesPredictionModel):
#     def __init__(self, base_sales_model: BaseSalesModel, uplift_model: SalesUpliftModel):
#         super().__init__(base_sales_model, uplift_model)
#
#     @property
#     def base_sales_model_info(self):
#         return self._base_sales_model.info
#
#     def get_sales(self, inputs: torch.Tensor, output_info: bool = False) -> \
#             Union[List[float], Tuple[List[float], Dict[str, Any]]]:
#
#         markdowns = inputs[:, 0].detach().cpu().numpy().flatten()
#         stocks = inputs[:, 2].detach().cpu().numpy().flatten()
#         base_sales = self._get_base_sales(inputs[:, 1:]).detach().cpu().numpy().flatten()
#         sales_uplifts = self._get_uplift(markdowns)
#         sales, info = [], {"base_sales": [], "sales_uplift": []}
#         for base_sale, sales_uplift, markdown in zip(base_sales, sales_uplifts, markdowns):
#             sales.append(stocks, base_sales * sales_uplift)
#             # log info
#             info["base_sales"].append(base_sale)
#             info["sales_uplift"].append(sales_uplift)
#
#         if output_info:
#             return sales, info
#         else:
#             return sales
