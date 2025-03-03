# Solar_Panel_Detection
# Solar Panel Detection using YOLOv8

## Project Overview
This repository focuses on detecting solar panels in aerial imagery using the YOLOv8 object detection model. The dataset consists of high-resolution (HD) and native-resolution images, with corresponding annotations. The goal is to preprocess the dataset, train the model, and evaluate its performance.

## Dataset Structure
```
Dataset/
├── hd_resized/         # Resized HD images (416x416)
├── native_resized/     # Resized Native images (416x416)

Labels/
├── labels_hd/         # YOLO-formatted annotations for HD images
├── labels_native/     # YOLO-formatted annotations for Native images
```

## Notebooks & Scripts
```
notebooks/
├── data_exploration.ipynb       # Exploratory Data Analysis (EDA) on images and labels
├── data_preprocessing.ipynb     # Image and label preprocessing scripts
├── function_implementation.ipynb # Helper functions for dataset handling
├── model_building.ipynb         # YOLOv8 model training and evaluation

scripts/
├── preprocess.py      # Python script for image resizing and format conversion
├── train.py           # Training script for YOLOv8
├── evaluate.py        # Evaluation script for metrics computation

results/
├── predictions/       # Model predictions on test images
├── logs/              # Training logs and loss curves
```

## Dataset Preparation
- Two sets of images: **HD Resized** and **Native Resized**, both resized to 416x416.
- Each image has a corresponding YOLO annotation file in `Labels/labels_hd/` and `Labels/labels_native/`.
- Dataset split: **80% training, 10% validation, and 10% test**.

## Preprocessing Steps
- **Image conversion**: Converting images to JPEG format.
- **Resizing**: Rescaling images to 416x416 for YOLOv8 compatibility.
- **Train-test split**: Creating an 80-10-10 split for training, validation, and testing.

## Model Training
- **Model**: YOLOv8 is used for object detection.
- **Training**: The model is trained on both HD and Native datasets.
- **Loss Monitoring**: Ensuring validation loss convergence.
- **Augmentation**: Applying data augmentation techniques like flipping, rotation, and color adjustments.

## Evaluation Metrics
- **Mean Average Precision (mAP@50)**: Evaluated using `supervision.metrics`.
- **Confusion Matrix**: Used to compute Precision, Recall, and F1-score.
- **Threshold Analysis**: Metrics computed for IoU thresholds [0.1, 0.3, 0.5, 0.7, 0.9].

## Visualization
- **Bounding Boxes**: Comparing ground truth vs. predicted bounding boxes.
- **Test Sample Predictions**: Displaying 3-4 random test images with detections.


## Dependencies
- Python 3.10
- ultralytics (for YOLOv8)
- OpenCV,NumPy,Pandas
- supervision(for evaluation metrics)(refer to the actual documentation of supervision,a lot has changed)

## Acknowledgments
This project is part of an internship and also personal interest
s.

