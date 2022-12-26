# SAD-SLAM: Sign-Agnostic Dynamic SLAM


* Course: 3D Computer Vision with Deep Learning Applications 
* Semester: Fall 2022
* Final Project: Sign-Agnostic Dynamic Simultaneous Localization and Mapping (SAD-SLAM)
* Group 11
* Group Members: 王琮文、詹易玹、王奕方
* Instructor: Chu-Song Chen
* National Taiwan University

---

## NICE-SLAM: Neural Implicit Scalable Encoding for SLAM
In 2022, Zhu et al. proposed NICE-SLAM that incorporates multi-level local information, 
and introducing neural implicit representations 
makes the dense SLAM system more scalable, efficient, and robust.
However, NICE-SLAM still has some issues presented in our report. 
To improve the orignal NICE-SLAM architecture, 
we propose Sign-Agnostic Dynamic SLAM (SAD-SLAM). 
The objective of our project is to 
* optimize mapping and tracking, and
* remove dynamic objects.
<br />
We validate our proposed SAD-SLAM on some datasets and experiments. 
Please refer to the three individual directories,  
thank you. 

---

## Sign-Agnostic (SA) Optimization
We improved the performance of mapping and tracking in NICE SLAM using Sign-Agnostic optimization.
Please change the working directory to `./SA_optimization` and read the `README.md` file in the folder for more information.

---

## Dynamic Objects Removal
We implemented dynamic objects removal in NICE SLAM using Mask R-CNN and background inpainting.
Please change the working directory to `./dynamic_objects_removal` and read the `README.md` file in the folder for more information.

---

## Experiments
To validate NICE-SLAM and our proposed methods, we used Intel® RealSense™ Depth Camera D435i, an RGB-D camera, to compare the results. 
Please change the working directory to `./experiment` and read the `README.md` file in the folder for more information.
