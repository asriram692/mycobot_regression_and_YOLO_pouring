"""
Plot spatial positioning accuracy comparing YOLO-Geometric baseline offsets 
vs. Direct Coordinate Regression under lower lighting conditions.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROBOT_BASE_COORD = (0.0, 40.0)
TARGET_RADIUS_MM = 25.5

# Ground Truth Centers
gt_coords = {
    "Pos 1 (Top-L)": (192.2, 139.7),
    "Pos 2 (Top-R)": (331.8, 139.7),
    "Pos 4 (Mid-L)": (242.89, 12.7),
    "Pos 5 (Mid-R)": (351.01, 12.7),
    "Pos 6 (Bot-L)": (193.67, -193.67),
    "Pos 7 (Bot-R)": (318.77, -193.67),
}

# Recorded offsets from center for Hardcoded baseline (Lower Lighting)
hardcoded_table_offsets = {
    "Pos 2 (Top-R)": (7, 7.83),
    "Pos 4 (Mid-L)": (6.8, 2.1),
    "Pos 5 (Mid-R)": (-6.63, 10),
    "Pos 6 (Bot-L)": (-12.2, 4.5),
    "Pos 7 (Bot-R)": (-13.9, 4.16),
}

# Recorded offsets from center for Neural Network trials (Lower Lighting)
nn_table_offsets = {
    "Pos 1 (Top-L)": [(33, 11.6), (32, 11.5), (8.5, 18.2)],
    "Pos 2 (Top-R)": [(2.2, 3.3), (2, 2.3), (1.5, 5.8)],
    "Pos 4 (Mid-L)": [(34, 14), (18.5, 18), (24, 10)],
    "Pos 5 (Mid-R)": [(6.2, 18.3), (2.2, 1), (2, 0)],
    "Pos 6 (Bot-L)": [(15.2, 17.3), (16.2, 21.7), (16.4, 16.2)],
    "Pos 7 (Bot-R)": [(3.2, 9.5), (5, 5), (11.05, -15.49)],
}

fig, ax = plt.subplots(figsize=(10, 7.5))
ax.scatter(
    ROBOT_BASE_COORD[0],
    ROBOT_BASE_COORD[1],
    color="black",
    marker="s",
    s=220,
    label=f"Robot Base {ROBOT_BASE_COORD}",
    zorder=6,
)

hc_success_legend_added = False
hc_fail_legend_added = False
nn_success_legend_added = False
nn_fail_legend_added = False

for idx, (spot_label, (gt_x, gt_y)) in enumerate(gt_coords.items()):
  tolerance_zone = patches.Circle(
      (gt_x, gt_y),
      radius=TARGET_RADIUS_MM,
      edgecolor="black",
      facecolor="lightgray",
      linestyle="--",
      linewidth=1.2,
      alpha=0.35,
      zorder=2,
      label=f"{TARGET_RADIUS_MM}mm Acceptance Zone" if idx == 0 else "",
  )
  ax.add_patch(tolerance_zone)
  ax.scatter(
      gt_x,
      gt_y,
      color="black",
      marker="X",
      s=130,
      zorder=5,
      label="Ground Truth Center" if idx == 0 else "",
  )
  ax.text(
      gt_x - 12,
      gt_y - 20,
      spot_label,
      fontsize=8.5,
      fontweight="bold",
      color="black",
  )

  if spot_label in hardcoded_table_offsets:
    hc_dx, hc_dy = hardcoded_table_offsets[spot_label]
    hc_land_x, hc_land_y = gt_x + hc_dx, gt_y + hc_dy
    hc_dist = np.sqrt(hc_dx**2 + hc_dy**2)

    hc_color = "darkturquoise" if hc_dist <= TARGET_RADIUS_MM else "deeppink"
    hc_label_text = (
        "YOLO-Baseline Mean (Success <= 25.5mm)"
        if hc_dist <= TARGET_RADIUS_MM
        else "YOLO-Baseline Mean (Fail > 25.5mm)"
    )

    if hc_dist <= TARGET_RADIUS_MM and not hc_success_legend_added:
      current_label = hc_label_text
      hc_success_legend_added = True
    elif hc_dist > TARGET_RADIUS_MM and not hc_fail_legend_added:
      current_label = hc_label_text
      hc_fail_legend_added = True
    else:
      current_label = ""

    ax.scatter(
        hc_land_x,
        hc_land_y,
        color=hc_color,
        marker="s",
        s=80,
        zorder=4,
        label=current_label,
    )
    ax.plot(
        [gt_x, hc_land_x],
        [gt_y, hc_land_y],
        color=hc_color,
        linestyle="-",
        linewidth=1.2,
        alpha=0.7,
    )
  else:
    ax.text(
        gt_x - 22,
        gt_y + 28,
        "(No YOLO-Baseline Data)",
        fontsize=7.5,
        fontstyle="italic",
        color="deeppink",
        fontweight="bold",
    )

  if spot_label in nn_table_offsets:
    offsets = np.array(nn_table_offsets[spot_label])
    nn_land_x = gt_x + offsets[:, 0]
    nn_land_y = gt_y + offsets[:, 1]

    mean_nn_x, mean_nn_y = np.mean(nn_land_x), np.mean(nn_land_y)
    mean_dist = np.sqrt((mean_nn_x - gt_x) ** 2 + (mean_nn_y - gt_y) ** 2)

    ax.scatter(
        nn_land_x,
        nn_land_y,
        color="orange",
        alpha=0.6,
        s=35,
        marker="o",
        zorder=3,
        label="Individual Regression Model Runs" if idx == 0 else "",
    )
    nn_color = "green" if mean_dist <= TARGET_RADIUS_MM else "crimson"
    nn_label_text = (
        "Regression Model Mean (Success <= 25.5mm)"
        if mean_dist <= TARGET_RADIUS_MM
        else "Regression Model Mean (Fail > 25.5mm)"
    )

    if mean_dist <= TARGET_RADIUS_MM and not nn_success_legend_added:
      nn_current_label = nn_label_text
      nn_success_legend_added = True
    elif mean_dist > TARGET_RADIUS_MM and not nn_fail_legend_added:
      nn_current_label = nn_label_text
      nn_fail_legend_added = True
    else:
      nn_current_label = ""

    ax.scatter(
        mean_nn_x,
        mean_nn_y,
        color=nn_color,
        marker="D",
        s=110,
        zorder=5,
        edgecolors="k",
        linewidth=0.5,
        label=nn_current_label,
    )
    ax.plot(
        [gt_x, mean_nn_x],
        [gt_y, mean_nn_y],
        color=nn_color,
        linestyle=":",
        linewidth=1.5,
        alpha=0.85,
    )

ax.set_title("Lower Lighting Pour Offsets", fontsize=14, fontweight="bold")
ax.set_xlabel("Physical Workspace X (mm)", fontsize=11)
ax.set_ylabel("Physical Workspace Y (mm)", fontsize=11)
ax.set_xlim(-30, 400)
ax.set_ylim(-240, 200)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle=":", alpha=0.5)

# Push the legend down safely using subplots_adjust and bbox_to_anchor
plt.subplots_adjust(bottom=0.25)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.32),
    ncol=2,
    fontsize=8.5,
    framealpha=0.9,
)

plt.savefig("lower_lighting_offsets.png", dpi=300, bbox_inches="tight")
plt.show()