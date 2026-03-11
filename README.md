# Intelligent Adaptive Traffic Flow Optimization System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

## Overview

Intelligent Adaptive Traffic Flow Optimization System is a deep learning-driven traffic management project designed to improve urban intersection efficiency and emergency response time. The system performs real-time object detection and applies post-detection adaptive signal logic to optimize vehicle flow, prioritize emergency vehicles, and enforce pedestrian-aware crossing behavior.

## Problem Statement

Urban traffic control systems are traditionally fixed-cycle and non-adaptive to real-time road conditions - varying traffic densities, pedestrian activity, and emergency vehicles. This causes inefficient traffic handling, longer waiting times, and delays in emergency response.

## Objectives

- Detect and classify traffic participants (vehicles, pedestrians, emergency vehicles) in real time
- Dynamically adjust signal timing based on detections
- Prioritize emergency vehicles for faster clearance
- Ensure pedestrian safety automatically

## Architecture and Methodology

### Detection Pipeline

- Fine-tuned YOLOv8m model with 16 classes for real-time object detection
- Trained on a merged 4-source dataset in YOLO format
- COCO-style labels extended with emergency vehicle classes starting from index 80

### Signal Control Logic

- If active emergency vehicle detected -> immediate green light priority
- If pedestrian count >= 5 -> trigger pedestrian crossing signal
- Otherwise -> balanced green duration proportional to vehicle count

### Inference Flow

1. Input traffic scene image
2. Run YOLOv8m detection
3. Count vehicles, pedestrians, emergency status
4. Apply rule-based decision layer
5. Output signal recommendation and dashboard summary

## Datasets

1. Cityscapes Dataset - Urban street scenes with fine annotations (dashcam perspective)
2. Smart Traffic Light System.v7i (Roboflow) - Emergency vehicle detection with active/non-active classification
3. Pedestrian Dataset - People detection at intersections in diverse urban settings
4. City Intersection Dataset - Top-angle crowded intersection views common in Indian cities

All datasets were merged, converted to YOLO format, and labels were aligned for unified multi-class training.

## Model Training Details

- Architecture: YOLOv8m (medium variant, speed-accuracy balance)
- Number of classes: 16
- Training epochs: 50
- Batch size: 16
- Initial learning rate: 0.001 (cosine decay)
- Early stopping: enabled with patience
- Backbone freezing: freeze=10
- Input size: 640 x 640

### Preprocessing and Augmentation

- YOLO format annotation conversion
- Image normalization
- Min-max label scaling
- Horizontal flip augmentation
- Brightness augmentation

## Results and Metrics

### Overall Validation Metrics

| Metric | Value |
|---|---:|
| mAP@0.5 | 92.7% |
| mAP@0.5:0.95 | 75.6% |
| Precision | 92.4% |
| Recall | 87.9% |
| F1 Score | 89.7% |
| Accuracy | 75.1% |
| Inference Speed | ~23 ms/image |

### Class-wise Performance Highlights

| Class | Precision | Recall | mAP@0.5 |
|---|---:|---:|---:|
| firetruck_active | 0.997 | 1.000 | 0.995 |
| ambulance_active | 0.946 | 0.986 | 0.977 |
| ambulance_nonactive | 0.972 | 0.917 | 0.977 |
| car | 0.966 | 0.886 | 0.953 |
| person | 0.810 | 0.465 | 0.601 |

Note: Person-class recall is lower due to frequent occlusion in crowded scenes.

## Key Features

- Real-time multi-class object detection using YOLOv8
- Emergency vehicle prioritization (ambulance, fire truck, police car; active vs non-active)
- Pedestrian-aware signal control (pedestrian trigger at >= 5 detections)
- Dynamic green-light duration estimation based on vehicle density
- Bounding box visualization with confidence scores
- Signal dashboard output: vehicle count, pedestrian count, emergency status, and suggested green duration

## Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- Roboflow
- Cityscapes Dataset
- Matplotlib

## Prerequisites

Before running the project, make sure you have:

- Python 3.9 or higher
- `pip` package manager
- A local copy of this project containing `app.py`, `best.pt`, `requirements.txt`, and `README.md`

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

After startup, open the Streamlit URL shown in terminal (typically `http://localhost:8501`).

## Example Inference Usage

```bash
streamlit run app.py
```

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("sample_intersection.jpg", conf=0.25)
results[0].show()
```

## Project Structure

```text
Intelligent-Adaptive-Traffic-Flow-Optimization-System/
├── app.py
├── best.pt
├── requirements.txt
├── README.md
└── test_images/
    ├── sample1.jpg
    └── sample2.jpg
```

## Limitations

- Performance drops under heavy occlusion (vehicles/pedestrians behind trucks or tightly grouped)
- Pedestrian recall is lower (46.5%) due to occlusion and dataset constraints
- Slight degradation in low-light/night conditions without IR support

## Future Scope

1. Integrate DeepSORT for real-time tracking plus GPS-based emergency preemption
2. Deploy on edge devices (NVIDIA Jetson Nano or Google Coral TPU)
3. Integrate Reinforcement Learning with SUMO for fully adaptive control
4. Add multimodal sensing (siren audio, radar, LiDAR)
5. Improve nighttime robustness using low-light preprocessing and IR-trained models
