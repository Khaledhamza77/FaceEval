# FaceEval

## Overview
The FaceEval Python package is designed to evaluate face image quality using traditional Computer Vision techniques. It uses RetinaFace model from **insightface** library, specifically the "buffalo_l" model. Using OpenCV and insightface, this package is able to provide quick and accurate evaluation of the face image quality. The upside of using this package is explainability and customizability, where every single threshold and parameter can be edited by the user. This can be demanding but I have already applied strict parameters such that this package can avoid saying that a good image is bad. I reccomend trying it out on a test sample of your own data and carrying out a parameter optimization experiment. Anyway, let's get into the details.

## Features
This algorithm performs the following:
1. **Face Detection with Bounding Box and Key Points**: Using insightface's buffalo_l detector (which is a retinaface model), this package can detect faces in images with GPU-acceleration
2. **Face Shape Quality Checks**: This package evaluates the bounding box and key points to find obvious quality issues for face images these problems are as follows:
    - Bounding box to image ratio: The bounding box cannot be too small or too large in relation to the image size.
    - Image coverage of the bounding box: The bounding box cannot be largerly outside of the image from both x and y axes.
    - Pose analysis: Key points cover both eyes, nose, and both lip corners. Based on the relationships between those points unacceptable poses are flagged. These can be vertical or horizontal.
3. **Face Lighting Quality Checks**: This package evaluates the lighting on a holistic level and on the level of specefic features derived from key points. By creating an intrinsic image as well as creating a Gaussian Image Pyramid depending on the image size and with a specific algorithm for evaluating brightness and occlusion, this package can detect quality problems on the level of facial features. More information about this will be shared in an article regarding this methodology on my website: www.khaledibrahim.site/ai-blog.
4. **Face Count and Image Corruption Handling**: This package is production-ready given the level of logging and error handling it has. It can handle corrputed images and flags images with more than one face.
5. **Parallelizable Implementation**: The FaceEvaluator class can take a device_id parameter which can enable running over multiple GPUs.

## Installation
To get started with TunedLLM, you can install it directly from this GitHub repository using pip.
1. Ensure Git is installed on your system
2. Install faceEval using the following command:
```bash
pip install git+https://github.com/Khaledhamza77/FaceEval
```

## Setup
In the production environment where this is working, this package is able utilize GPU-acceleration using onnxruntime-gpu with the specs of this AWS Deep Learning Container: pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker. Further dependecies are included in the requirements file. If there any issues with setup or with getting insightface to utilize GPU using CUDAExecutionProvider, please contact me and I can share with you my conda environment yaml file which covers all other dependencies.

## Demo

## Package Structure

## Contact Information
For any inquiries or potential contribution please contact me at: khaledhamza@aucegypt.edu