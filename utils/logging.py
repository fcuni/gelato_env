from typing import Any, Dict, List, Optional
from datetime import datetime
from collections import defaultdict

import pandas as pd
import torch
import plotly.express as px
from plotly.subplots import make_subplots


class EpisodeLogger:

    def __init__(self):
        self.stock_per_product: Dict[str, List[int]] = defaultdict(list)
        self.sales_per_product: Dict[str, List[int]] = defaultdict(list)
        self.revenue_per_product: Dict[str, List[float]] = defaultdict(list)
        self.base_sales_per_product: Dict[str, List[float]] = defaultdict(list)
        self.uplift_per_product: Dict[str, List[float]] = defaultdict(list)
        self.current_price_per_product: Dict[str, List[float]] = defaultdict(list)
        self.current_markdowns_per_product: Dict[str, List[float]] = defaultdict(list)
        self.sales_not_clipped_per_product: Dict[str, List[float]] = defaultdict(list)
        self.dates: List[datetime] = []
        self.products: List[str] = []
        self.flavours: List[str] = []
        self.step: int = 0

    def log_info(self, info: Dict[str, Any]):
        if "current_date" in info:
            self.dates.append(info["current_date"])

        if len(self.products) == 0:
            self.products = [product_id for product_id in info["products"]]
            self.flavours = [info["flavours"][product_id] for product_id in info["products"]]

        for i, product_id in enumerate(info["products"]):

            # if i == 0:
            #     self.products.append(product_id)
            #     if "flavours" in info:
            #         flavour = info["flavours"][product_id] if isinstance(info["flavours"], dict) else info["flavours"][i]
            #         self.flavours.append(flavour)

            if "stocks" in info:
                stock = info["stocks"][product_id] if isinstance(info["stocks"], dict) else info["stocks"][i]
                if isinstance(stock, torch.Tensor):
                    stock = stock.item()
                self.stock_per_product[product_id].append(int(stock))

            if "sales" in info:
                sales = info["sales"][product_id] if isinstance(info["sales"], dict) else info["sales"][i]
                if isinstance(sales, torch.Tensor):
                    sales = sales.item()
                self.sales_per_product[product_id].append(int(sales))
            else:
                self.sales_per_product[product_id].append(0)

            if "current_price" in info:
                current_price = info["current_price"][product_id] if isinstance(info["current_price"], dict) else \
                    info["current_price"][i]
                if isinstance(current_price, torch.Tensor):
                    current_price = current_price.item()
                self.current_price_per_product[product_id].append(current_price)
                if "sales" in info:
                    revenue = max(0.0, current_price * sales)
                    self.revenue_per_product[product_id].append(revenue)
                else:
                    self.revenue_per_product[product_id].append(0.0)

            if "base_sales" in info:
                base_sales = info["base_sales"][product_id] if isinstance(info["base_sales"], dict) else \
                    info["base_sales"][i]
                if isinstance(base_sales, torch.Tensor):
                    base_sales = base_sales.item()
                self.base_sales_per_product[product_id].append(base_sales)
            else:
                self.base_sales_per_product[product_id].append(0.0)

            if "sales_uplift" in info:
                uplift = info["sales_uplift"][product_id] if isinstance(info["sales_uplift"], dict) else \
                    info["sales_uplift"][i]
                if isinstance(uplift, torch.Tensor):
                    uplift = uplift.item()
                self.uplift_per_product[product_id].append(uplift)
            else:
                self.uplift_per_product[product_id].append(1.0)

            if "sales_not_clipped" in info:
                sales_not_clipped = info["sales_not_clipped"][product_id] if isinstance(info["sales_not_clipped"],
                                                                                        dict) else \
                    info["sales_not_clipped"][i]
                if isinstance(sales_not_clipped, torch.Tensor):
                    sales_not_clipped = sales_not_clipped.item()
                self.sales_not_clipped_per_product[product_id].append(sales_not_clipped)
            else:
                self.sales_not_clipped_per_product[product_id].append(0.0)

            if "current_markdowns" in info:
                current_markdown = info["current_markdowns"][product_id] if isinstance(info["current_markdowns"],
                                                                                       dict) else \
                    info["current_markdowns"][i]
                if isinstance(current_markdown, torch.Tensor):
                    current_markdown = current_markdown.item()
                self.current_markdowns_per_product[product_id].append(current_markdown)

        self.step += 1

    def save(self):
        """Save the logged information to a pickle file."""
        import pickle
        import os
        import datetime

        if len(self.products) == 0:
            return

        if not os.path.exists("logs"):
            os.makedirs("logs")

        date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"logs/{date}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def plot_by_date(self, plot_type: str = "sales"):
        """Plot the logged information."""

        if plot_type == "sales":
            record = self.sales_per_product
        elif plot_type == "stock":
            record = self.stock_per_product
        elif plot_type == "revenue":
            record = self.revenue_per_product
        elif plot_type == "base_sales":
            record = self.base_sales_per_product
        elif plot_type == "uplift":
            record = self.uplift_per_product
        elif plot_type == "current_price":
            record = self.current_price_per_product
        elif plot_type == "current_markdowns":
            record = self.current_markdowns_per_product
        elif plot_type == "sales_not_clipped":
            record = self.sales_not_clipped_per_product
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        # df = pd.DataFrame(data=[], columns=['Date', *self.flavours])
        # df = pd.concat([df, pd.DataFrame({'Date': [date for date in self.dates], **record})])
        df = pd.DataFrame({'Date': [date for date in self.dates], **record})
        df.columns = ['Date', *self.flavours]
        fig = px.line(df, x='Date', y=self.flavours)
        return fig

    def plot_episode_summary(self, title: Optional[str] = None):

        # Plot the episode summary
        plot_types = ("current_markdowns", "revenue", "stock", "sales")
        plot_titles = ("Actions taken", "Revenue", "Stock", "Sales")
        figs = []
        for plot_type in plot_types:
            # plot the single figure of each plot type
            single_fig = self.plot_by_date(plot_type=plot_type)
            # disable the legend of each plot
            single_fig.update_layout(showlegend=False)
            # add the single figure to the list of figures
            figs.append(single_fig)

        # Create a 2x2 subplot layout
        fig = make_subplots(rows=2, cols=2, shared_xaxes="all", vertical_spacing=0.15, horizontal_spacing=0.05,
                            subplot_titles=plot_titles,
                            x_title="Date")

        # Add each of the original figures to the subplots
        for i, subplot_fig in enumerate(figs):
            for trace in subplot_fig['data']:

                # disable the legend of each trace
                if i != 0:
                    trace.update(showlegend=False)
                fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)

        # Add the title to the figure
        if title is not None:
            fig.update_layout(
                title_text=title,
                title_x=0.5,  # Set the title's x position to be centered
                title_font_size=24,  # Adjust the font size of the title
            )

        return fig

    def get_episode_summary(self, key_prefix: Optional[str] = None) -> Dict[str, Any]:
        """Return the per-product summary as a dictionary."""

        summary = {
            "sales_by_product": {},
            "base_sales_by_product": {},
            "uplift_sales_by_product": {},
            "revenue_by_product": {},
            "remaining_stock_by_product": {},

            "start_date": self.dates[0].strftime("%Y-%m-%d"), # date of init state (first date to take action)
            "end_date": self.dates[-1].strftime("%Y-%m-%d"),  # date of final state (the resulting state of the final action)
            "total_steps": self.step - 1  # number of steps taken
        }

        for i, product_id in enumerate(self.products):
            summary["sales_by_product"][self.flavours[i]] = sum(self.sales_per_product[product_id])
            summary["base_sales_by_product"][self.flavours[i]] = sum(self.base_sales_per_product[product_id])
            summary["uplift_sales_by_product"][self.flavours[i]] = summary["sales_by_product"][self.flavours[i]] \
                - summary["base_sales_by_product"][self.flavours[i]]  # the difference between total sales and base sales
            summary["revenue_by_product"][self.flavours[i]] = sum(self.revenue_per_product[product_id])
            summary["remaining_stock_by_product"][self.flavours[i]] = self.stock_per_product[product_id][-1]

        summary["total_sales"] = sum(summary["sales_by_product"].values())
        summary["total_base_sales"] = sum(summary["base_sales_by_product"].values())
        summary["total_uplift_sales"] = sum(summary["uplift_sales_by_product"].values())
        summary["total_revenue"] = sum(summary["revenue_by_product"].values())
        summary["total_remaining_stock"] = sum(summary["remaining_stock_by_product"].values())

        # add the prefix to the keys if specified
        if key_prefix is not None:
            summary = {f"{key_prefix}{key}": value for key, value in summary.items()}

        return summary
