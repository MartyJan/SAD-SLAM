# Dynamic Objects Removal
## Environment Setup

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `nice-slam`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate nice-slam
```


## Dataset
The [Bonn RGB-D Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) was used to evaluate the performance of our model in dynamic scenes. We selected the image sequence named “rgbd_bonn_crowd” in the dataset, which captures a scene sometimes blocked by people walking through, for model evaluation. Download this testing image sequence from [here](https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019/rgbd_bonn_crowd.zip).


## Background Inpainting Demo
Change the working directory to the `demo` folder and run `repaint.py`:
```bash
cd ./demo
python repaint.py
```
This outputs six images `prev_*.png`, `curr_*.png` and `repainted_*.png` in the working directory. `prev_rgb.png` and `prev_depth.png` are the RGB-D channel of the source frame (previous keyframe) for background inpainting. `curr_rgb.png` and `curr_depth.png` are the RGB-D channel of the current frame to be repainted. `repainted_rgb.png` and `repainted_depth.png` are the RGB-D channel of the current frame after background inpainting. 

## NICE-SLAM with Mask R-CNN

First, unzip the `rgbd_bonn_crowd.zip` file into the `./segment/Datasets/` directory.
```bash
unzip rgbd_bonn_crowd -d .segment/Datasets/
```

Second, change the working directory to the `segment` folder and run SLAM.
```bash
cd ./segment
python -W ignore run.py configs/Bonn/rgbd_bonn_crowd.yaml
```

Then, run the following command to visualize scene mesh reconstruction.
```bash
python visualizer.py configs/Bonn/rgbd_bonn_crowd.yaml
```

To evaluate the average trajectory error, run the command below.
```bash
python src/tools/eval_ate.py configs/Bonn/rgbd_bonn_crowd.yaml
```
This will output the ATE plot stored as `eval_ate_plot.png` in the `./segment/output/rgbd_bonn_crowd/` directory.


## NICE-SLAM with Mask R-CNN and Background Inpainting

First, unzip the `rgbd_bonn_crowd.zip` file into the `./segment_repaint/Datasets/` directory.
```bash
unzip rgbd_bonn_crowd -d .segment_repaint/Datasets/
```

Second, change the working directory to the `segment_repaint` folder and run SLAM.
```bash
cd ./segment_repaint
python -W ignore run.py configs/Bonn/rgbd_bonn_crowd.yaml
```
Note that this program requires a lot of computational resources. It processed only 1/10 of our testing image sequence for about 5hr when running on Nvidia RTX 3080.

Then, run the following command to visualize scene mesh reconstruction.
```bash
python visualizer.py configs/Bonn/rgbd_bonn_crowd.yaml
```