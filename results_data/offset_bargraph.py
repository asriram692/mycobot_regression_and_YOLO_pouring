"""Plot average positioning offset across environmental conditions for YOLO Baseline vs. Direct Coordinate Regression."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

environments = ["Ambient", "Darker", "Cardboard"]

nn_offsets = [18.78, 23.72, 30.61]
hardcode_offsets = [15.17, 12.20, 22.40]

x = np.arange(len(environments))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))

rects1 = ax.bar(
    x - width / 2,
    hardcode_offsets,
    width,
    label="YOLO Baseline",
    color="#4C72B0",
)
rects2 = ax.bar(
    x + width / 2,
    nn_offsets,
    width,
    label="Regression Model",
    color="#DD8452",
)

ax.set_ylabel("Average Spatial Offset (mm)", fontsize=12, fontweight="bold")
ax.set_title(
    "Model Performance Comparison Across Environments",
    fontsize=14,
    fontweight="bold",
)
ax.set_xticks(x)
ax.set_xticklabels(environments, fontsize=11)

ax.axhline(
    y=25.5,
    color="red",
    linestyle="--",
    linewidth=1.5,
    label="Acceptance Zone Limit (25.5 mm)",
)
ax.legend(loc="upper left", fontsize=11)

ax.bar_label(rects1, padding=3, fmt="%.1f mm")
ax.bar_label(rects2, padding=3, fmt="%.1f mm")

fig.tight_layout()

plt.savefig("assets/offset_bar.png", dpi=300, bbox_inches="tight")
plt.show()