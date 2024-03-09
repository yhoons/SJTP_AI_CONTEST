# SEJONG AI CONTEST

<img width="876" alt="image" src="https://github.com/yhoons/SJTP_AI_CONTEST/assets/79200729/875ba38a-e95f-412b-ba7b-7795105b03c0">

## Introduction

AI Contest to Detect Jaywalking Pedestrians Using Road CCTV Data with YOLOV8

![sejongTP drawio](https://github.com/yhoons/SJTP_AI_CONTEST/assets/79200729/e51825ee-7e47-4070-bc07-1e09bf11d8dd)

### Requirements
```
roboflow==1.1.19
opencv==4.8.0
numpy==1.26.2
ultralytics==8.0.196
```

### Evaluation
```
python jaywalk_detectv2.py --data_dir ./imgset --result_dir ./result
```
--data_dir : input own data file
--resultdir : output own data file


## Result
Write and store label and normalized pedestrian bounding box coordinate values in a text file

![results](https://github.com/yhoons/SJTP_AI_CONTEST/assets/79200729/6004cbf0-6f93-4f7c-a9ea-af07f88519a0)
<img width="553" alt="image" src="https://github.com/yhoons/SJTP_AI_CONTEST/assets/79200729/e8298f83-7826-422a-a553-5bf260d4dee8">
<img width="553" alt="image" src="https://github.com/yhoons/SJTP_AI_CONTEST/assets/79200729/44d87625-c348-4798-97bb-e1889ce2a36f">

result label
label 0 : Jaywalking Pedestrian
label 1 : Not Jaywalking Pedestrian

