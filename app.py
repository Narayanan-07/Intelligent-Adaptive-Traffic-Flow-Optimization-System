import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from PIL import Image
import os
from collections import Counter

# Load YOLO model
model = YOLO("runs/detect/final model/weights/best.pt")

st.title("🚦 Smart Traffic Signal Controller")
st.markdown("Upload a traffic scene image. The model will predict objects and calculate signal duration based on detections.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    frame = np.array(image)

    results = model.predict(frame, conf=0.25)
    result = results[0]

    # Display detection result with bounding boxes
    st.image(result.plot(), caption="YOLOv8 Detections", use_column_width=True)

    # Count objects
    vehicle_count = 0
    pedestrian_count = 0
    emergency_present = False
    class_counter = Counter()

    for box in result.boxes.data:
        x1, y1, x2, y2, conf, cls = box.cpu().numpy()
        label = model.names[int(cls)]
        class_counter[label] += 1

        if label in ["car", "truck", "bus", "motorcycle", "ambulance_nonactive", "police_nonactive", "firetruck_nonactive"]:
            vehicle_count += 1
        elif label == "person":
            pedestrian_count += 1
        elif label in ["ambulance_active", "firetruck_active", "police_active"]:
            emergency_present = True

    # Adjust signal logic
    if pedestrian_count>=0 and vehicle_count==0 and emergency_present==False:
        if pedestrian_count>0:
            ped_green = max(int(pedestrian_count * 0.9), 6)
            signal_status = f"🚶 Pedestrian Green Light: {ped_green}s 🚗 Vehicle: RED"
        else:
            signal_status = "No signal : Orange"
    elif pedestrian_count >= 5:
        if pedestrian_count <= 20:
            ped_green = max(int(pedestrian_count * 0.9), 6)
        else:
            ped_green = 20
        signal_status = f"🚶 Pedestrian Green Light: {ped_green}s 🚗 Vehicle: RED"
    else:
        if vehicle_count >0 and vehicle_count<= 5:
            base_green = vehicle_count + 2
        elif vehicle_count <= 10:
            base_green = int(vehicle_count * 1.2)
        elif vehicle_count <= 20:
            base_green = int(vehicle_count * 1.5)
        else:
            base_green = int(vehicle_count * 1.8)

        base_green = min(base_green, 55)

        if emergency_present:
            adjusted_time = min(base_green + 10, 60)
            signal_status = f"🚦 Vehicle Green Light (Emergency): {adjusted_time}s 🚶 Pedestrian: RED"
        else:
            signal_status = f"🚦 Vehicle Green Light: {base_green}s 🚶 Pedestrian: RED"

    st.subheader("🔢 Signal Adjustment Result")
    st.success(f"**{signal_status}**")

    # Display object counts
    st.subheader("📅 Detection Summary")
    st.write(f"👥 Pedestrians: {pedestrian_count}")
    st.write(f"🚗 Vehicles: {vehicle_count}")
    st.write(f"🚨 Emergency Detected: {'Yes' if emergency_present else 'No'}")

    # Class Distribution Plot
    st.subheader("📊 Class Distribution")
    labels = list(class_counter.keys())
    values = list(class_counter.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

    # Confusion Matrix Section
   