# Analytical-Crowd-Counter
AI based application that counts the number of people in a crowd and also detects the total social distancing violations dynamically.

## Introduction

The analytical crowd counter is a tool for crowd control that would analyze a video, detect how many people are present in any particular scene and also detect the number of social distancing violations occurring. 

Crowd counting refers to estimating the number of individuals who share a certain region. The analytical crowd counter investigates the behavior of a large group of people sharing the same physical area. Typically, it counts the number of individuals per region, tracks the common individuals’ trajectories, and recognizes individuals’ behaviors. It also supplies real time surveillance camera systems with the ability to extract anomalous behaviors from a huge group of people.

The Analytical crowd counter uses OpenCV, computer vision, and deep learning for detecting and making analysis on the video file provided.

<center><img src="https://user-images.githubusercontent.com/67074796/123330530-c348cb00-d55b-11eb-9240-c619724b2c7e.png"  width="500" ></center>

#### Methodology involved:
*	Using the YOLO object detector to detect people in a video stream
*	Determining the centroids for each detected person
*	Computing the number of people present in the video (crowd) and pairwise distances between all centroids
*	Checking to see if any pairwise distances were < than the defined value, and if so, indicating that the pair of people violated social distancing rules
*	Furthermore, by using an NVIDIA CUDA-capable GPU, along with OpenCV’s DNN module compiled with NVIDIA GPU support, our method will be able to run in real-time, making it usable   as a proof-of-concept crowd counter and social distancing detector.

#### Sample Screnshot


<img src="https://user-images.githubusercontent.com/67074796/123329668-b081c680-d55a-11eb-8b77-79f9c7ab0946.png" width="650" >

#### Acticity Diagram

<img src="https://user-images.githubusercontent.com/67074796/123330324-8250b680-d55b-11eb-9fc0-49094b090ac8.jpg" width="650" >

