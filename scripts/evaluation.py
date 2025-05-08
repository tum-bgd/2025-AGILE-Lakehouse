# %%
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt

EXP = "01_chunk_half"

# NAME = "200M"
# PATH_LAZ = "../data/AHN4/C_69AZ1.LAZ"
# PATH_PARQUET = "../data/AHN4/C_69AZ1_convert.parquet"
# PATH_GRID = "../data/AHN4/C_69AZ1_grid(1).parquet"
# PATH_GRID_8 = "../data/AHN4/C_69AZ1_grid(8).parquet"
# PATH_QUADTREE = "../data/AHN4/C_69AZ1_quadtree.parquet"

NAME = "2B"
PATH_LAZ = "../data/AHN3/C_37E*.LAZ"
PATH_PARQUET = "../data/AHN3/C_37E*_convert.parquet"
PATH_GRID = "../data/AHN3/C_37E*_grid(1).parquet"
PATH_GRID_8 = "../data/AHN3/C_37E*_grid(8).parquet"
PATH_QUADTREE = "../data/AHN3/C_37E*_quadtree.parquet"

DATASETS = {
    "convert": {"path": PATH_PARQUET, "color": "tab:brown", "marker": "o"},
    "grid(1)": {"path": PATH_GRID, "color": "tab:olive", "marker": "s"},
    "grid(8)": {"path": PATH_GRID_8, "color": "tab:orange", "marker": "+"},
    "quadtree": {"path": PATH_QUADTREE, "color": "tab:purple", "marker": "^"},
}

# --------------------------------------------------------------------------- #
# Collect metrics
# --------------------------------------------------------------------------- #
for k, v in DATASETS.items():

    # stats
    stats = {
        "rows": 0,
        "rg": 0,
        "points": [],
        "delta_x": [],
        "delta_y": [],
        "delta_i": [],
    }

    paths = glob.glob(v["path"])

    for path in paths:
        # metadata
        meta = pq.read_metadata(path)

        stats["rows"] += meta.num_rows
        stats["rg"] += meta.num_row_groups

        for i in range(meta.num_row_groups):
            row_group = meta.row_group(i)
            stats["points"].append(row_group.num_rows)

            for ii in range(meta.num_columns):
                column = row_group.column(ii)

                statistics = column.statistics

                match column.path_in_schema:
                    case "x":
                        stats["delta_x"].append(statistics.max - statistics.min)
                    case "y":
                        stats["delta_y"].append(statistics.max - statistics.min)
                    case "i":
                        if isinstance(statistics.max, bytes):
                            stats["delta_i"].append(
                                np.frombuffer(statistics.max, dtype=np.float16)[0]
                                - np.frombuffer(statistics.min, dtype=np.float16)[0]
                            )
                        else:
                            stats["delta_i"].append(statistics.max - statistics.min)

    stats["points"] = np.array(stats["points"])
    stats["delta_x"] = np.array(stats["delta_x"])
    stats["delta_y"] = np.array(stats["delta_y"])
    stats["delta_i"] = np.array(stats["delta_i"])
    stats["area"] = stats["delta_x"] * stats["delta_y"]
    stats["volume"] = stats["area"] * stats["delta_i"]

    v |= stats


# --------------------------------------------------------------------------- #
# Descriptive stats
# --------------------------------------------------------------------------- #
# %%
for k, v in DATASETS.items():
    print(
        "FILES: {} ({}), ROWS: {}, RG: {}".format(
            v["path"], len(paths), v["rows"], v["rg"]
        )
    )

    for s in ["points: ", "delta_x:", "delta_y:", "delta_i:", "area:   ", "volume: "]:
        a = s.strip(": ")
        print("{} {:12.2f} ({:12.2f})".format(s, v[a].mean(), v[a].std()))


# --------------------------------------------------------------------------- #
# Mean row groups extent
# --------------------------------------------------------------------------- #
# %%
fig, axs = plt.subplots(3, figsize=(8, 4.5))

ATTRS = ["delta_x", "delta_y", "delta_i"]
LABELS = ["mean x range [m]", "mean y range [m]", "mean i range"]

y_pos = np.arange(len(DATASETS))
colors = [v["color"] for v in DATASETS.values()]

for i, attr in enumerate(ATTRS):
    mean = [ds[attr].mean() for ds in DATASETS.values()]
    std = [ds[attr].std() for ds in DATASETS.values()]
    # axs[i].bar(mean, width, color=colors)  # , yerr=stds)

    axs[i].barh(y_pos, mean, color=colors, xerr=std, align="center")
    axs[i].set_yticks(y_pos, labels=DATASETS.keys())
    axs[i].invert_yaxis()  # labels read top-to-bottom
    axs[i].set_xlabel(LABELS[i])

    axs[i].spines[["right", "top"]].set_visible(False)

fig.tight_layout()
plt.savefig(f"../logs/rg_extent_{NAME}_{EXP}.jpg", dpi=300)
plt.show()


# --------------------------------------------------------------------------- #
# Row Group Intersection
# --------------------------------------------------------------------------- #
# %%
df = pd.read_csv(f"../logs/rg_{NAME}.csv")

N = df["level"].nunique()

ind = np.arange(N)  # the x locations for the groups
width = 0.2  # the width of the bars

fig = plt.figure(figsize=(8, 4.5))


ax = fig.add_subplot(111)

grids = []
for i, (k, v) in enumerate(DATASETS.items()):
    f = df["file"] == k

    mean = df[f]["mean"]
    std = df[f]["std"]
    grid = ax.bar(df[f]["level"] + i * width, mean, width, color=v["color"], yerr=std)
    grids.append(grid)

# add some
ax.set_ylabel("Number of Row Groups (mean)")
ax.set_xticks(ind + width * 1.5)
ax.set_xticklabels([f"Level {i}" for i in ind])

ax.legend(grids, DATASETS.keys(), frameon=False)

ax.spines[["right", "top"]].set_visible(False)

fig.tight_layout()
plt.savefig(f"../logs/rg_{NAME}_{EXP}.jpg", dpi=300)
plt.show()


# --------------------------------------------------------------------------- #
# Visualization query
# --------------------------------------------------------------------------- #
# %%
df = pd.read_csv(f"../logs/viz_{NAME}.csv")

N = df["level"].nunique()

data = []
data_indexed = []
for level in range(N):
    f = df["level"] == level
    data.append(df[f]["time"])
    data_indexed.append(df[f]["time_indexed"])

fig, ax = plt.subplots(figsize=(7, 3.5))

# datafusion
pos = np.arange(N) - 0.2

bp_df = ax.boxplot(
    data, positions=pos, widths=0.3, patch_artist=True, manage_ticks=False
)

for median in bp_df["medians"]:
    median.set_color(DATASETS["quadtree"]["color"])
for box in bp_df["boxes"]:
    box.set_facecolor("white")

plt.axhline(
    y=np.nanmean(df["time"].mean()),
    linestyle="--",
    color=DATASETS["quadtree"]["color"],
    label="DataFusion",
    zorder=-1.0,
)

# indexed
pos = np.arange(N) + 0.2

bp_index = ax.boxplot(
    data_indexed, positions=pos, widths=0.3, patch_artist=True, manage_ticks=False
)

for median in bp_index["medians"]:
    median.set_color("tab:pink")
for box in bp_index["boxes"]:
    box.set_facecolor("white")

plt.axhline(
    y=np.nanmean(df["time_indexed"].mean()),
    linestyle="--",
    color="tab:pink",
    label="Indexed",
    zorder=-1.0,
)

ax.set_ylabel("Query response time [ms]")
plt.xticks(range(N))
ax.set_xticklabels([f"Level {i}" for i in range(N)])
ax.spines[["right", "top"]].set_visible(False)
ax.legend(
    [bp_df["medians"][0], bp_index["medians"][0]],
    ["DataFusion", "Index"],
    loc="upper right",
    frameon=False,
)

fig.tight_layout()
plt.savefig(f"../logs/viz_{NAME}_{EXP}.jpg", dpi=300)
plt.show()

# %%
