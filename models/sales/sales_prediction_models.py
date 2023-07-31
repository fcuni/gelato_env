from typing import Sequence, List

import torch

from models.sales.sales_uplift_models import SalesUpliftModel
from models.sales.base_sales_models import BaseSalesModel


class SalesPredictionModel:
    def __init__(self, base_sales_model: BaseSalesModel, uplift_model: SalesUpliftModel):
        self._base_sales_model = base_sales_model
        self._uplift_model = uplift_model

    def _get_base_sales(self, inputs: torch.Tensor):
        return self._base_sales_model.get_sales(inputs)

    def _get_uplift(self, markdown: Sequence[float]):
        return self._uplift_model(markdown)

    def get_sales(self, inputs, markdowns: Sequence[float]) -> List[float]:
        assert len(inputs) == len(markdowns)
        sales = []
        for input_data, markdown in zip(inputs, markdowns):
            stock = input_data[1] # TODO: to identify stock column
            base_sales = self._get_base_sales(input_data)
            sales_uplift = self.get_uplift(markdown)
            sales.append(min(stock, base_sales * sales_uplift))
        return sales
