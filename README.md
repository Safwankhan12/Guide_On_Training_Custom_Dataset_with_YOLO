# Training Custom Dataset with Ultralytics YOLO
![ultralytics_yolov8_image](https://github.com/Safwankhan12/Guide_On_Training_Custom_Dataset_with_YOLO/assets/128811059/f4ee28c5-c134-4b24-aaa8-74fff59145b9)

## Introduction

The YOLO (You Only Look Once) algorithm is a pioneering approach to object detection in computer vision. Unlike traditional methods, YOLO treats object detection as a single regression problem, directly predicting bounding boxes and class probabilities for objects in an image. This streamlined approach results in impressive speed and accuracy, making YOLO a popular choice for real-time applications like autonomous driving and surveillance. Its simplicity and efficiency have revolutionized object detection, making YOLO a key tool in advanced computer vision systems.

## Yolo Modes and Tasks 
### Tasks
YOLO includes four core tasks: detection, segmentation, classification, and pose estimation.

1) Detection identifies objects.  


2) Segmentation partitions images. 


3) Classification categorizes objects. 


4) Pose estimation determines object poses.

Example : yolo detect train data=coco8.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640

### Modes
YOLO comprises three primary modes:

Train: In this mode you train the YOLOv model using your custom dataset.

Validate: Here, you validate the performance of your trained model.

Predict: In the final phase, you utilize your trained model to make predictions on new data.


## Custom vs Pre-trained model


With YOLOv, you have two choices: either employ a pre-trained model with the extension .pt, or train your own model with your dataset, requiring the extension .yaml.

### Pre-trained model
When using a pre-trained model, you don't need to provide any dataset for training. The model is already trained; you just need to provide images for it to make predictions on.

!yolo task=detect mode=predict model=yolov8n.pt conf=0.25 source=test/images

### Custom trained model
For custom training a model, you first need a dataset to train it. You can acquire datasets from platforms like <a href='https://universe.roboflow.com/'>Roboflow</a> or  <a href='https://kaggle.com/datasets'>Kaggle</a>

If your data is unannotated, the first step is to annotate it using a tool like <a href='https://www.cvat.ai/'>CVAT</a>. Once annotated, you can then use it to train your model.


#### Steps

1) Upload your train test and val folder which you downloaded from Roboflow, kaggle or custom annotated to google drive

2) Make a new colaboratory file inside of your drive

3) Make sure your data.yaml file is also uploaded to drive and is in the same directory as colaboratory file. This will be used to locate your train, test and valid folders

4) Open colaboratory and mount your google drive to colaboratory. Check your current path using !ls

5) Install ultralytics using !pip install ultralytics

6) In data.yaml file change the path of test val and train to test: /content/drive/MyDrive/your_folder/test/images, /content/drive/MyDrive/your_folder/valid/images, /content/drive/MyDrive/your_folder/train/images respectively

7) Start training of your model using command !yolo task=detect mode=train model=yolov8s.yaml data=data.yaml epochs=3

8) Now you're going to have best.pt trained model in runs/detect/train/weights folder

9) Validate your model using command !yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=data.yaml

10) Finally your model is ready to make predcitions. To predict using your model use !yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=data/test/images

## Dataset used in this Tutorial
<a href='https://universe.roboflow.com/school-vakkx/mango-fruit-detection/dataset/6'>Fruit Dataset<a/>
## Ultralytics Documentation

https://github.com/ultralytics/ultralytics?tab=readme-ov-file


https://docs.ultralytics.com/
