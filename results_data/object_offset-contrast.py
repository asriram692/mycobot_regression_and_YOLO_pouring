"""
Plot spatial positioning accuracy comparing YOLO-Geometric baseline offsets 
vs. Direct Coordinate Regression under contrast distraction conditions.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROBOT_BASE_COORD = (0.0, 40.0)
TARGET_RADIUS_MM = 25.5

gt_coords = {
    "Pos 1 (Top-L)": (192.2, 139.7),
    "Pos 2 (Top-R)": (331.8, 139.7),
    "Pos 4 (Mid-L)": (242.89, 12.7),
    "Pos 5 (Mid-R)": (351.01, 12.7),
    "Pos 6 (Bot-L)": (193.67, -193.67),
    "Pos 7 (Bot-R)": (318.77, -193.67),
}

hardcoded_table_offsets = {
    "Pos 1 (Top-L)": (36.8, 3.8),
    "Pos 2 (Top-R)": (15.5, 17),
    "Pos 4 (Mid-L)": (24.1, -12.1),
    "Pos 5 (Mid-R)": (5.8, -1.5),
    "Pos 6 (Bot-L)": (-20.3, 5.4),
    "Pos 7 (Bot-R)": (-13.8, 2.4),
}

nn_table_offsets = {
    "Pos 1 (Top-L)": [(28.6, 9.1), (14.31, 3.8), (26.8, 3.8)],
    "Pos 2 (Top-R)": [(4.1, 27.7), (19, 14.6), (26.2, 6.5)],
    "Pos 4 (Mid-L)": [(41.4, 8.2), (26.5, 34.4), (50.6, 12)],
    "Pos 5 (Mid-R)": [(30.216, 36), (38.2, 35.3), (29, 33.1)],
    "Pos 6 (Bot-L)": [(52.2, 10), (7.7, 2.1), (25.8, 3.1)],
    "Pos 7 (Bot-R)": [(3.12, 22.8), (0, 7), (8, 6)],
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

ax.set_title(
    "Contrast Distraction Pour Offsets", fontsize=14, fontweight="bold"
)
ax.set_xlabel("Physical Workspace X (mm)", fontsize=11)
ax.set_ylabel("Physical Workspace Y (mm)", fontsize=11)
ax.set_xlim(-30, 400)
ax.set_ylim(-240, 200)

ax.set_aspect("equal", adjustable="box")
ax.grid(True, linestyle=":", alpha=0.5)

plt.subplots_adjust(bottom=0.25)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, -0.32),
    ncol=2,
    fontsize=8.5,
    framealpha=0.9,
)

plt.savefig("contrast_distraction_offsets.png", dpi=300, bbox_inches="tight")
plt.show()