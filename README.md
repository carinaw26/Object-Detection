# Object Detection

## Tutorial
How to do object detection with pre-trained models for image file, video file and live video.

Reference https://www.machinecurve.com/index.php/2021/01/15/object-detection-for-images-and-videos-with-tensorflow-2-x/

### jupyter notebook
```code
cd jupyter-notebook/
jupyter notebook
```
Select object_detection_study.ipynb
Following the instruction

### Run step by step Python script

#### Prerequisites
##### Python packages
```code
pip install tensorflow #This will install numpy and related packges
pip install opencv-python
```
##### Protoc
Download pre-built protoc. For Windows download protoc-3.5.0-win32.zip and unzip it. Add the bin directory to PATH environment variable.

https://github.com/protocolbuffers/protobuf/releases/tag/v3.5.0

##### Object Detection Repository
```code
git clone https://github.com/tensorflow/models

```
##### Download pre-trained model
Download TensorFlow 2 Detection Model from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
Consider Spped (ms) and COCO (Microsoft Common Object in Context database) mAP (mean Average Precision)
For exmample,
- CenterNet HourGlass104 512x512 (70 ms 41.9 COCO mAP)
- ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8 (39 ms, 28.2 COCO mAP)
- EfficientDet D1 640x640 ( 54 ms 38.4 COCO mAP)
- Faster R-CNN ResNet101 V1 640x640 (55ms 31.8 mAP)

Download the mode and unpack it. You will set the directory to ot_model when create TFObjectDetector instance.

#### Run the object detection script
```code
python Tutorial/ah_object_detection_1.py
```
Click 'q' to stop.