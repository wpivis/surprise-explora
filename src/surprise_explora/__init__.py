import importlib.metadata
import pathlib
import numpy as np
import scipy.stats as stats
import altair as alt
import pandas as pd
import anywidget
import traitlets

alt.data_transformers.disable_max_rows()
# alt.data_transformers.enable("vegafusion")

try:
    __version__ = importlib.metadata.version("surprise_explora")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


class Widget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "static" / "widget.js"
    _css = pathlib.Path(__file__).parent / "static" / "widget.css"
    value = traitlets.Int(0).tag(sync=True)


def cdf(s):
    return stats.norm.cdf(s, 0, 1)


class SurpriseGroup:
    def __init__(self, df, name, rate_key, population_key):
        self.df = df
        self.name = name
        self.rate_key = rate_key
        self.population_key = population_key

        self.rate_mean = self.df[self.rate_key].mean()
        self.std_dev = self.df[self.rate_key].std()

    def calculate(self):
        # calculate the z-score
        # data[fipsCode].zScore = (+data[fipsCode].rate - rateMean) / rateStdDev;
        self.zScore = (self.df[self.rate_key] - self.rate_mean) / self.std_dev

        # calculate likelihood (pMs)
        test_statistic = (self.df[self.rate_key] - self.rate_mean) / (
            self.std_dev
            / np.sqrt(self.df[self.population_key] / self.df[self.population_key].sum())
        )

        pMs = 2 * (1 - cdf(abs(test_statistic)))
        self.pMs = np.maximum(pMs, 1e-10)

        # calculate kl divergence
        self.kl = self.pMs * np.log(self.pMs) / np.log(2)

        # calculate surprise
        surprise = np.where(
            self.df[self.rate_key] - self.rate_mean > 0,
            np.abs(self.kl),
            -1 * np.abs(self.kl),
        )
        self.surprise = np.where(self.df[self.rate_key] == 0, 0, surprise)

        # put z-score back into the dataframe
        self.df[self.name + "_zScore"] = self.zScore

        # put surprise back into the dataframe
        self.df[self.name + "_surprise"] = self.surprise


class Surprise:
    def __init__(
        self,
        df,
        global_rate_key: str,
        global_population_key: str,
        groups=[],
        rate_keys=[],
        population_keys=[],
    ):
        self.df = df
        self.groups = groups
        self.rate_keys = rate_keys
        self.population_keys = population_keys
        self.global_rate_key = global_rate_key
        self.global_population_key = global_population_key

        self.surprise_groups = []
        for idx, group in enumerate(self.groups):
            self.surprise_groups.append(
                SurpriseGroup(
                    self.df, group, self.rate_keys[idx], self.population_keys[idx]
                )
            )

        self.global_group = SurpriseGroup(
            self.df, "global", self.global_rate_key, self.global_population_key
        )

    def calculate(self):
        for group in self.surprise_groups:
            group.calculate()

        self.surprise_keys = [
            group.name + "_surprise" for group in self.surprise_groups
        ]

        # print(self.global_group)
        self.global_group.calculate()
        self.df["global_surprise"] = self.global_group.surprise

    def bar_chart(self, state: str):
        self.surprise_keys.append("global_surprise")
        average_rate = self.df["global_rate"].mean()
        df_long = self.df.melt(
            id_vars=["name", "state"],
            value_vars=self.surprise_keys,
            var_name="Group",
            value_name="Surprise",
        )

        df_filtered = df_long[df_long["state"] == state]      

        bar_chart = (
            alt.Chart()
            .mark_bar(size=15)
            .encode(
                x=alt.X("Group:N"),
                y=alt.Y("Surprise:Q").scale(domain=[-0.125, 0.125]),
                color="Group:N",
                tooltip=[
                alt.Tooltip("Surprise:Q", format=".3f"),
                alt.Tooltip("Group:N", title="Race Group"),
                ],
            )
            .transform_filter(alt.datum.Group != "global_surprise")
            .properties(width=200, height=200)
        )

        line_chart = (
            alt.Chart()
            .mark_rule(color="black", strokeWidth=1, strokeDash=[2, 2])  
            .encode(
                y=alt.Y("Surprise:Q")
            ).transform_filter(alt.datum.Group == "global_surprise")
        )
        
        layered_chart = (bar_chart + line_chart).facet(facet="name:N", columns=5, data=df_filtered).interactive()

        return layered_chart
