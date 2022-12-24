# Experiments for NICE-SLAM

## Installation
First you have to make sure that you have all dependencies in place. 
You can create an anaconda environment called ```sad-slam-exp```  . 
For linux, you need to install libopenexr-dev before creating the environment.
```  
sudo apt-get install libopenexr-dev
conda create --name sad-slam-exp --file spec.txt
conda activate sad-slam-exp
```
## Parameters (e.g., file paths)
You could modify your input and output paths in the file ``` configs/realsense.yaml```. 
This github repo provides one set of experimental data stored at 
```Datasets/Realsense/scene0```. 
The corresponding result from NICE-SLAM is stored at ```output/Realsense/scene0```; 
while the result from the SA optimization is stored at ```output/Realsense/scene0_SA```. 

## Running the SLAM algorithm
Our experimental data has been stored in the folder ```Datasets``` and ```output```. 
The folder ```Datasets``` includes the original RGB images and depth information. 
The folder ```output``` includes the camera poses and mesh files from the output of the SLAM algorithm. 
Besides, we use Intel RealSense depth camera D435i to generate RGB images and depth information. 
```  
python -W ignore run.py configs/realsense.yaml
```

## Visualizing results
```  
python visualizer.py configs/realsense.yaml --no_gt_traj
```

![niceslam_mesh](output/Realsense/scene0/chair_niceslam.gif)
