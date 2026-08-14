"""Plot direct regression coordinate predictions vs. ground truth targets across three environmental conditions."""
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# DATA DEFINITIONS
# ---------------------------------------------------------

# 1. Ambient Control Data
gt_coords_ambient = {
    "Pos 1 (Top-L)": (93.4, 120.6),
    "Pos 2 (Top-R)": (119.7, 138.5),
    "Pos 4 (Mid-L)": (123.1, 33.8),
    "Pos 5 (Mid-R)": (162.1, 32.5),
    "Pos 6 (Bot-L)": (128.0, -65.4),
    "Pos 7 (Bot-R)": (191.7, -97.3)
}

nn_preds_ambient = {
    "Pos 1 (Top-L)": [(106.0, 139.7), (112.2, 132.3), (110.3, 137.4)],
    "Pos 2 (Top-R)": [(118.7, 141.7), (121.0, 151.5), (122.8, 148.4)],
    "Pos 4 (Mid-L)": [(119.6, 32.8), (126.1, 57.7), (125.7, 42.6)],
    "Pos 5 (Mid-R)": [(160.6, 39.0), (167.0, 57.5), (177.0, 46.5)],
    "Pos 6 (Bot-L)": [(118.6, -47.1), (115.9, -60.1), (109.5, -48.1), (112.9, -50.1), (114.8, -48.0)],
    "Pos 7 (Bot-R)": [(183.6, -80.5), (183.4, -76.3), (190.6, -67.9)]
}

# 2. Lower Lighting Data
gt_coords_lowlight = {
    "Pos 1 (Top-L)": (93.4, 118.6),
    "Pos 2 (Top-R)": (119.7, 138.5),
    "Pos 4 (Mid-L)": (123.1, 33.8),
    "Pos 5 (Mid-R)": (162.1, 32.5),
    "Pos 6 (Bot-L)": (128.0, -65.4),
    "Pos 7 (Bot-R)": (191.7, -97.3)
}

nn_preds_lowlight = {
    "Pos 1 (Top-L)": [(93.1, 133.5), (89.5, 124.3), (89.5, 130.9)],
    "Pos 2 (Top-R)": [(145.7, 131.0), (143.2, 128.1), (140.8, 124.3)],
    "Pos 4 (Mid-L)": [(119.5, 43.7), (119.2, 49.2), (120.4, 43.8)],
    "Pos 5 (Mid-R)": [(159.1, 40.5), (164.1, 32.6), (164.8, 33.2)],
    "Pos 6 (Bot-L)": [(109.5, -48.1), (112.9, -50.1), (114.8, -48.0)],
    "Pos 7 (Bot-R)": [(196.7, -87.0), (189.5, -67.0), (177.8, -88.4)]
}

# 3. High Contrast / Clutter Data
gt_coords_contrast = {
    "Pos 1 (Top-L)": (93.4, 118.6),
    "Pos 2 (Top-R)": (119.7, 138.5),
    "Pos 4 (Mid-L)": (123.1, 33.8),
    "Pos 5 (Mid-R)": (162.1, 32.5),
    "Pos 6 (Bot-L)": (128.0, -65.4),
    "Pos 7 (Bot-R)": (191.7, -97.3)
}

nn_preds_contrast = {
    "Pos 1 (Top-L)": [(89.8, 95.3), (103.1, 58.8), (103.3, 134.7)],
    "Pos 2 (Top-R)": [(142.2, 152.2), (142.6, 152.1), (162.1, 147.4)],
    "Pos 4 (Mid-L)": [(130.8, 54.1), (135.0, 56.7), (133.8, 54.6)],
    "Pos 5 (Mid-R)": [(176.4, 30.8), (176.1, 41.4), (174.6, 46.5)],
    "Pos 6 (Bot-L)": [(131.6, -42.4), (133.5, -42.5), (138.5, -40.1)],
    "Pos 7 (Bot-R)": [(188.5, -89.9), (188.8, -81.7), (188.1, -75.8)]
}

# Plot Configuration Setup
conditions = [
    ("Ambient Lighting (Control)", gt_coords_ambient, nn_preds_ambient, 'dodgerblue'),
    ("Lower Lighting", gt_coords_lowlight, nn_preds_lowlight, 'purple'),
    ("High Contrast Distraction", gt_coords_contrast, nn_preds_contrast, 'teal')
]



fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)

fig.suptitle("Regression Model Target Coordinate Prediction Accuracy Across Conditions", 
             fontsize=16, fontweight='bold', y=0.98)


for ax_idx, (title, gt_coords, nn_predictions, mean_color) in enumerate(conditions):
    ax = axes[ax_idx]
    
    for spot_idx, (spot_label, (gt_x, gt_y)) in enumerate(gt_coords.items()):
        

        ax.scatter(gt_x, gt_y, color='black', marker='X', s=120, zorder=5, 
                   label='Ground Truth Target' if (ax_idx == 0 and spot_idx == 0) else "")
        

        ax.text(gt_x - 12, gt_y - 12, spot_label, fontsize=8, fontweight='bold', color='black')

     
        if spot_label in nn_predictions:
            preds = np.array(nn_predictions[spot_label])
            pred_x, pred_y = preds[:, 0], preds[:, 1]
            
            mean_x, mean_y = np.mean(pred_x), np.mean(pred_y)
            mae_dist = np.sqrt((mean_x - gt_x)**2 + (mean_y - gt_y)**2)


            ax.scatter(pred_x, pred_y, color='darkorange', alpha=0.6, s=35, marker='o', zorder=3, 
                       label='Individual Trial Predictions' if (ax_idx == 0 and spot_idx == 0) else "")
            

            ax.scatter(mean_x, mean_y, color=mean_color, marker='D', s=100, zorder=6, 
                       edgecolors='black', linewidth=0.6, 
                       label='Mean NN Prediction' if (ax_idx == 0 and spot_idx == 0) else "")
            
            # Error Vector Line
            ax.plot([gt_x, mean_x], [gt_y, mean_y], color='crimson', linestyle='--', linewidth=1.4, alpha=0.8,
                    label='Perception Error Vector' if (ax_idx == 0 and spot_idx == 0) else "")
            
            ax.text((gt_x + mean_x)/2 + 2, (gt_y + mean_y)/2, f"{mae_dist:.1f} mm", 
                    fontsize=8, color='crimson', fontweight='bold')


    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("Camera-Inferred X Position (mm)", fontsize=10)
    if ax_idx == 0:
        ax.set_ylabel("Camera-Inferred Y Position (mm)", fontsize=10)

    ax.set_xlim(40, 220)
    ax.set_ylim(-130, 180)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.6)

#Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10, framealpha=0.9)

plt.subplots_adjust(bottom=0.15, top=0.88, wspace=0.15)
plt.savefig("combined_regression_positioning_accuracy.png", dpi=300, bbox_inches='tight')
plt.show()