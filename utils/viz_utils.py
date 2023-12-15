"""Most common visualization functions, using matplotlib and seaborn library

Most functions take in a dictionary of data with the following information
data = {
    "save_filepath": filepath to save figure, 
    "title": name of figure,
    "xlabel": str of x label,
    "ylabel: str of y label,

    # the rest of keys are customized, depending on plot types (line, bar, etc.)
    # check each function for details
}

Some tutorials for other plots
- matplotlib: https://matplotlib.org/stable/tutorials/index
- seaborn: https://seaborn.pydata.org/tutorial/introduction.html
"""
from collections import defaultdict

import os
import json
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

PALETTE = "deep"  # change here: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
# COLORS = itertools.cycle(sns.color_palette(palette=PALETTE, as_cmap=True))
COLORS = itertools.cycle(sns.color_palette(palette=PALETTE))
MARKER = itertools.cycle(("v", "+", ".", "o", "*", "^", "8"))


def scatterplot(data: dict, variance: bool = False):
    """Plot a basic scatterplot (no connecting line)
    TODO: merge this with scatterplot_with_line

    ## Args:
        data (dict): dictionary of data for plotting, important keys include
            data = {
                "x_data": list of x,
                "y_data": list of (y_name, y_value) data,
                "y_data_var": variance of y_data, also dict of (y_name, y_var),
                "style": {
                    "xticks": list of customized x ticks,
                    "yticks": list of customized y ticks,
                }
            }
        variance (bool): whether to include variance in graph

    ### example usage
    scatterplot_with_line(
        data={
            "save_filepath": "./hello.pdf",
            "title": "Model Accuracy",
            "xlabel": "Configs",
            "ylabel": "Acc",
            "x_data": list(range(5)),
            "y_data": [
                ("model 1", [0.1, 0.2, 0.3, 0.4, 0.5]),
                ("model 2", [0.4, 0.5, 0.6, 0.7, 0.8]),
            ],
            "y_data_var": [
                ("model 1", [0.2, 0.2, 0.2, 0.2, 0.2]),
                ("model 2", [0.1, 0.1, 0.1, 0.1, 0.1]),
            ],
            "style": {"fontsize": 12, "linestyle": "dotted"},
        },
        variance=True,
    )
    """
    # handle some style issue
    style = {
        "fontsize": 16,
        "linestyle": "dotted",  # better than "dashed" or "dashdot"
        "alpha": 0.2,  # for background hue
    }
    if "style" in data:
        for style_name, style_val in data["style"].items():
            style[style_name] = style_val

    # first plot main graph
    fig, ax = plt.subplots()
    handles = []
    for i, (y_name, y_val) in enumerate(data["y_data"]):
        ax.scatter(
            data["x_data"], y_val, label=y_name, marker=next(MARKER), color=next(COLORS)
        )

        # if variance
        if variance and "y_data_var" in data:
            ax.fill_between(
                data["x_data"],
                np.array(y_val) - np.array(data["y_data_var"][i][1]),
                np.array(y_val) + np.array(data["y_data_var"][i][1]),
                alpha=style["alpha"],
            )

    # set style
    ax.legend()
    ax.set_title(data["title"])
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.grid(linestyle=style["linestyle"])
    if "xticks" in style:
        ax.set_xticks(style["xticks"])
    if "yticks" in style:
        ax.set_yticks(style["yticks"])

    # finally save data
    fig.tight_layout()
    fig.savefig(data["save_filepath"])


def scatterplot_with_line(data: dict, variance: bool = False):
    """Plot a basic scatterplot with connecting line

    ## Args:
        data (dict): dictionary of data for plotting, important keys include
            data = {
                "x_data": list of x,
                "y_data": list of (y_name, y_value) data,
                "y_data_var": variance of y_data, also dict of (y_name, y_var),
                "style": {
                    "xticks": list of customized x ticks,
                    "yticks": list of customized y ticks,
                }
            }
        variance (bool): whether to include variance in graph

    ### example usage
    scatterplot_with_line(
        data={
            "save_filepath": "./hello.pdf",
            "title": "Model Accuracy",
            "xlabel": "Configs",
            "ylabel": "Acc",
            "x_data": list(range(5)),
            "y_data": [
                ("model 1", [0.1, 0.2, 0.3, 0.4, 0.5]),
                ("model 2", [0.4, 0.5, 0.6, 0.7, 0.8]),
            ],
            "y_data_var": [
                ("model 1", [0.2, 0.2, 0.2, 0.2, 0.2]),
                ("model 2", [0.1, 0.1, 0.1, 0.1, 0.1]),
            ],
            "style": {"fontsize": 12, "linestyle": "dotted"},
        },
        variance=True,
    )
    """
    # handle some style issue
    style = {
        "fontsize": 16,
        "linestyle": "dotted",  # better than "dashed" or "dashdot"
        "alpha": 0.2,  # for background hue
    }
    if "style" in data:
        for style_name, style_val in data["style"].items():
            style[style_name] = style_val

    # first plot main graph
    fig, ax = plt.subplots()
    handles = []
    for i, (y_name, y_val) in enumerate(data["y_data"]):
        (handle,) = ax.plot(
            data["x_data"], y_val, label=y_name, marker=next(MARKER), color=next(COLORS)
        )
        handles.append(handle)

        # if variance
        if variance and "y_data_var" in data:
            ax.fill_between(
                data["x_data"],
                np.array(y_val) - np.array(data["y_data_var"][i][1]),
                np.array(y_val) + np.array(data["y_data_var"][i][1]),
                alpha=style["alpha"],
            )

    # set style
    ax.legend(handles=handles)
    ax.set_title(data["title"])
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.grid(linestyle=style["linestyle"])
    if "xticks" in style:
        ax.set_xticks(style["xticks"])
    if "yticks" in style:
        ax.set_yticks(style["yticks"])

    # finally save data
    fig.tight_layout()
    fig.savefig(data["save_filepath"])


def bar_chart(data: dict, show_legend: bool = False):
    """Plot bar charts
    ### Args:
        data (dict): dictionary of data for plotting, important keys include
            data = {
                "x_data": list of x (should be categorical)
                "y_data": list of (y_name, y_value) data (len of y_value == len(x_data))
                "y_data_var": variance of y_data, also list of (y_name, y_var),
                "style": {
                    # these are to compute the xticks and bar locations
                    "bar_width": width of bars,
                    "bar_slack": gap between bars
                }
            }
    ### example usage
    bar_chart(
        data={
            "save_filepath": "./hello.pdf",
            "title": "Model Accuracy",
            "xlabel": "Configs",
            "ylabel": "Acc",
            "x_data": ["config 1", "config 2", "config 3"],
            "y_data": [
                ("model 1", [0.5, 0.5, 0.5]),
                ("model 2", [0.7, 0.7, 0.7]),
            ],
            "style": {
                "bar_width": 0.25,
                "bar_slack": 0.0,
                "bar_font": 10,
                "color": "blue", # for a single value (eg categorical distribution)
                "ylim": (0, 1),
                "figsize": [12, 8],
            },
        },
        show_legend=True,
    )
    """
    # first handle style variables
    style = {
        "bar_width": 0.25,
        "bar_slack": 0.01,
        "bar_font": 14,
        "bar_pad": 2,
        "fontsize": 14,
        "legend_loc": "upper right",
        "ylim": (0, 100),
        "figsize": [8, 6],
    }
    if "style" in data:
        for style_name, style_val in data["style"].items():
            style[style_name] = style_val
    plt.rcParams["figure.figsize"] = style["figsize"]
    plt.rcParams["font.size"] = style["fontsize"]

    # then plot
    fig, ax = plt.subplots()
    for col_idx, (y_name, y_val) in enumerate(data["y_data"]):

        # first set label locations
        y_loc = [
            cat + col_idx * (style["bar_width"] + style["bar_slack"])
            for cat in np.arange(len(data["x_data"]))
        ]
        bar = ax.bar(
            x=y_loc,
            height=y_val,
            width=style["bar_width"],
            label=y_name,
            color=next(COLORS) if "color" not in style else style["color"],
            edgecolor="black",
        )

        # also set bar label
        ax.bar_label(bar, fontsize=style["bar_font"], padding=style["bar_pad"])

    # set style
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title(data["title"])
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])
    ax.set_ylim(style["ylim"][0], style["ylim"][1])
    ax.tick_params(axis="both", labelsize=style["fontsize"])
    ax.set_yticklabels
    num_cols = len(data["y_data"])
    ax.set_xticks(
        np.arange(len(data["x_data"]))
        + (num_cols - 1) / (num_cols) * style["bar_width"]
        + (num_cols) * style["bar_slack"]
    )
    ax.set_xticklabels(data["x_data"])
    if show_legend:
        ax.legend(loc=style["legend_loc"], fontsize=style["fontsize"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # finally save data
    fig.tight_layout()
    fig.savefig(data["save_filepath"])


def distribution(data: dict):
    """Plotting distribution from data
    ###Args:
        data (dict): dictionary of data for plotting, important keys include
            data = {
                "x_data": list of x (should be categorical)
                "y_data": list of (y_name, y_value) data
                "bin_width": width of bins,
                "kde": kernel density estimation (smooth distribution)
            }


    ### example usage
    distribution(
        data={
            "save_filepath": "./hello.pdf",
            "title": "Data Distributions",
            "xlabel": "X",
            "ylabel": "Count",
            "hue_label": "Dataset",
            "y_data": [
                ("dataset 1", list(np.random.randint(0, 10, size=10))),
                ("dataset 2", list(np.random.randint(3, 13, size=10))),
                ("dataset 3", list(np.random.randint(7, 17, size=10))),
            ],
            "bin_width": 3,
            "kde": True,
        },
    )
    """
    # handle style stuff
    style = {"palette": "tab10"}
    if "style" in data:
        for style_name, style_val in data["style"].items():
            style[style_name] = style_val

    # construct dataframe from dictionary
    data_dict = {data["hue_label"]: [], data["xlabel"]: []}
    for y_name, y_val in data["y_data"]:
        num_points = len(y_val)
        data_dict[data["hue_label"]] += [y_name] * num_points
        data_dict[data["xlabel"]] += y_val
    data_pd = pd.DataFrame.from_dict(data_dict)

    # plot
    sns.displot(
        data=data_pd,
        x=data["xlabel"],
        hue=data["hue_label"],
        binwidth=data["bin_width"],
        kde=data["kde"],
        palette=style["palette"],
    )

    # save plot
    plt.savefig(data["save_filepath"])


def heatmap(data: dict, annotated: bool = False):
    """Plotting heatmap
    ### Args:
        data (dict): dictionary of data for plotting, important keys include
            data = {
                "data": 2D list of data for heatmap
                "xticks": list of x ticks
                "yticks": list of y ticks
            }
        annotated (bool): whether to annotate each box with value

    ### Example Usage TODO
    """
    # handle style stuff
    style = {"fontsize": 8, "figsize": [12, 8], "color_map": "BuGn", "shrink": 0.75}
    if "style" in data:
        for style_name, style_val in data["style"].items():
            style[style_name] = style_val
    plt.rcParams["font.size"] = style["fontsize"]  # set smaller font
    plt.rcParams["figure.figsize"] = style["figsize"]

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(data["data"], cmap=style["color_map"])
    cbar = ax.figure.colorbar(im, ax=ax, shrink=style["shrink"])
    cbar.ax.set_ylabel(data["sidebar_label"], rotation=0, va="bottom")

    if annotated:  # loop over data dimensions and create text annotations
        for i in range(len(data["data"])):
            for j in range(len(data["data"][0])):
                text = ax.text(
                    j,
                    i,
                    "%0.1f" % data["data"][i, j],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize="x-small",
                )

    # set style
    # show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(data["xticks"])), labels=data["xticks"])
    ax.set_yticks(np.arange(len(data["yticks"])), labels=data["yticks"])
    ax.set_title(data["title"])
    ax.set_xlabel(data["xlabel"])
    ax.set_ylabel(data["ylabel"])

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # save figure
    fig.savefig(data["save_filepath"])
