# Vision-Guided Autonomous Pouring: Direct Coordinate Regression vs. YOLO Baseline

An end-to-end computer vision and robotic manipulation pipeline developed for the **MyCobot 280** robotic arm. This project evaluates and compares two distinct perception approaches for target cup positioning and pouring execution.
1. **Direct Regression Model:** A MobileNetV3 model trained to directly map camera frames to target (X, Y) Cartesian workspace coordinates.
2. **YOLO Baseline:** A YOLOv8 object detection model combined with geometric center-point calculations and offsets.

---

##  Repository Structure

```text
mycobot_regression_and_YOLO_pouring/
├── models_data_collection/  # Scripts for dataset generation & model training
├── results_data/            # Scripts for evaluating spatial offsets & plotting
├── result_data_png/         # Generated output plots & performance charts
├── task_execution/          # Robot arm control & task runner pipelines
├── .gitignore               # Rules for ignoring model weights & cache
├── requirements.txt         # Project dependencies
└── README.md                # Documentation