# Sign-Agnostic (SA) Optimization
## Environment Setup

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 
You can create an anaconda environment called `nice-slam-sa`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate nice-slam-sa
```

## Dataset

We use ScanNet(500 frames) and Replica(room0) to evaluate the performance improvement of mapping and tracking.
First download the dataset.
### ScanNet
```bash
bash scripts/download_scannet.sh
```
### Replica
```bash
bash scripts/download_replica.sh
```

## Demo

Here we provide trained ScanNet and Replica for demo.
### Visualize
To visualize reconstructed mesh for ScanNet and Replica:
```bash
python visualizer.py configs/ScanNet/scannet.yaml
```
```bash
python visualizer.py configs/Replica/room0.yaml
```
### Tracking performance
To evaluate ATE and show the tracking trajectory for ScanNet and Replica:
```bash
python src/tools/eval_ate.py configs/ScanNet/scannet.yaml
```
```bash
python src/tools/eval_ate.py configs/Replica/room0.yaml
```
### Mapping performance
To evaluate reconstruction error on Accuracy, Completion, and Completion Ratio for Replica, first download the culled ground truth of Replica:
```bash
bash scripts/download_cull_replica_mesh.sh
```
Then run:
```bash
python src/tools/eval_recon.py --rec_mesh output/replica_sa/mesh/final_mesh_eval_rec.ply --gt_mesh cull_replica_mesh/room0.ply -2d -3d
```
## Training

Run the sa-optimization on NICE-SLAM for ScanNet and Replica:
```bash
python -W ignore run.py configs/ScanNet/scannet.yaml
```
```bash
python -W ignore run.py configs/Replica/room0.yaml
```















