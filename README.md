<p align="center">
<h1 align="center">
SAD-SLAM: Sign-Agnostic Dynamic SLAM
</h1>

<h2 align="center">
<a href="https://github.com/MartyJan/SAD-SLAM"><strong> code </strong></a>
|
<a href="https://youtu.be/dVUWtoyijMk"><strong> video </strong></a>
|
<a href="https://github.com/MartyJan/SAD-SLAM/blob/main/report.pdf"><strong> report </strong></a>
</h2>

</p>

---

# Final Project: SAD-SLAM
* Course: 3D Computer Vision with Deep Learning Applications 
* Semester: Fall 2022
* Topic: Sign-Agnostic Dynamic Simultaneous Localization and Mapping (SAD-SLAM)
* Group 11
* Group Members: 王琮文、詹易玹、王奕方
* Instructor: Chu-Song Chen
* National Taiwan University

---

## NICE-SLAM and SAD-SLAM
In 2022, Zhu et al. proposed Neural Implicit Scalable Encoding for SLAM (NICE-SLAM) 
that incorporates multi-level local information, 
and neural implicit representations are introduced to
make the dense SLAM system more scalable, efficient, and robust. 
However, NICE-SLAM still has some issues presented in our report. 
To improve the orignal NICE-SLAM architecture, 
we propose Sign-Agnostic Dynamic SLAM (SAD-SLAM). 
The objective of our project is to 
* optimize mapping and tracking, and
* remove dynamic objects. 
<br />
The results are demonstrated in our Youtube video and report: 
we validated our proposed SAD-SLAM on some datasets and experiments.
Three subtopics will be mainly discussed in the following sections, 
including two solutions and one experiment; 
they are all included in their individual directories. 

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

---

## References
[1] Zhu, Z., Peng, S., Larsson, V., Xu, W., Bao, H., Cui, Z., ... & Pollefeys, M. (2022). Nice-slam: Neural implicit scalable encoding for slam. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 12786-12796). <br />

[2] Dai, A., Chang, A. X., Savva, M., Halber, M., Funkhouser, T., & Nießner, M. (2017). Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5828-5839). <br />

[3] Straub, J., Whelan, T., Ma, L., Chen, Y., Wijmans, E., Green, S., ... & Newcombe, R. (2019). The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797. <br />

[4] Peng, S., Niemeyer, M., Mescheder, L., Pollefeys, M., & Geiger, A. (2020, August). Convolutional occupancy networks. In European Conference on Computer Vision (pp. 523-540). Springer, Cham. <br />

[5] Tang, J., Lei, J., Xu, D., Ma, F., Jia, K., & Zhang, L. (2021). Sa-convonet: Sign-agnostic optimization of convolutional occupancy networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 6504-6513). <br />

[6] Palazzolo, E., Behley, J., Lottes, P., Giguere, P., & Stachniss, C. (2019, November). Refusion: 3d reconstruction in dynamic environments for rgb-d cameras exploiting residuals. In 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (pp. 7855-7862). IEEE. <br />

[7] Bescos, B., Fácil, J. M., Civera, J., & Neira, J. (2018). DynaSLAM: Tracking, mapping, and inpainting in dynamic scenes. IEEE Robotics and Automation Letters, 3(4), 4076-4083. <br />

[8] He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 2961-2969). <br />


