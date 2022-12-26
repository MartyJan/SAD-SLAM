import os

import cv2
import numpy as np
import torch

from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from PIL import Image  
import math
import torch.nn.functional as F

H = 480
W = 640
fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
crop_edge = 8
crop_size = [384,512]
distortion = np.array([0.039903, -0.099343, -0.000730,  -0.000144, 0.000000])
png_depth_scale = 5000.0
scale = 1
device = "cpu"

def as_intrinsics_matrix(intrinsics):
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

def frame_reader(color_path, depth_path):
    color_data = cv2.imread(color_path)
    if '.png' in depth_path:
        depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if distortion is not None:
        K = as_intrinsics_matrix([fx, fy, cx, cy])
        # undistortion is only applied on color image, not depth!
        color_data = cv2.undistort(color_data, K, distortion)

    color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

    color_data = color_data / 255.
    depth_data = depth_data.astype(np.float32) / png_depth_scale
    H, W = depth_data.shape
    color_data = cv2.resize(color_data, (W, H))
    color_data = torch.from_numpy(color_data)
    depth_data = torch.from_numpy(depth_data)*scale
    if crop_size is not None:
        # follow the pre-processing step in lietorch, actually is resize
        color_data = color_data.permute(2, 0, 1)
        color_data = F.interpolate(
            color_data[None], crop_size, mode='bilinear', align_corners=True)[0]
        depth_data = F.interpolate(
            depth_data[None, None], crop_size, mode='nearest')[0, 0]
        color_data = color_data.permute(1, 2, 0).contiguous()

    edge = crop_edge
    if edge > 0:
        # crop image edge, there are invalid value on the edge of the color image
        color_data = color_data[edge:-edge, edge:-edge]
        depth_data = depth_data[edge:-edge, edge:-edge]
    # pose = self.poses[index]
    # pose[:3, 3] *= scale
    index = 0
    return index, color_data.to(device), depth_data.to(device)

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def unproject_points(uvs):
    K = as_intrinsics_matrix([fx, fy, cx, cy])
    return np.dot(np.linalg.inv(K), add_ones(uvs).T).T[:, 0:2] 

def rescale_translation_factor(kp_ref_u, kp_cur_u):
    kpn_ref = unproject_points(kp_ref_u)
    kpn_cur = unproject_points(kp_cur_u)
    
    mean_ref = np.mean(kpn_ref, axis=0)
    mean_cur = np.mean(kpn_cur, axis=0)
    ref_squared = 0.0
    cur_squared = 0.0
    
    for i in range(kpn_ref.shape[0]):
        kp_ref_u_0mean_i = kpn_ref[i] - mean_ref
        kp_cur_u_0mean_i = kpn_cur[i] - mean_cur
        ref_squared += np.dot(kp_ref_u_0mean_i, kp_ref_u_0mean_i)
        cur_squared += np.dot(kp_cur_u_0mean_i, kp_cur_u_0mean_i)
    rescale_factor = math.sqrt(cur_squared/ref_squared)
    rescale_factor = min(max(rescale_factor, 0.5), 2.0) # bound the scale in the range of [0.5 2.0]
    return rescale_factor

_, gt_color, gt_depth = frame_reader("./test_img/1548339829.02492.png", "./test_img/1548339828.96460.png")
cv2.imwrite('curr_rgb.png', gt_color.numpy()*255)
cv2.imwrite('curr_depth.png', gt_depth.numpy()*png_depth_scale)

""" Instance segmentation """
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()

model = maskrcnn_resnet50_fpn(weights=weights, progress=False)
model = model.eval()
curr_color_img = gt_color.detach().cpu().numpy()
curr_color_img = curr_color_img * 255.
curr_color_img = curr_color_img.astype(np.uint8)
pil_img = Image.fromarray(curr_color_img)
output = model([transforms(pil_img)])  # list
output = output[0]

masks = output['masks'].squeeze(1)
person_masks = []
for i in range(len(output['labels'])):
    if weights.meta["categories"][output['labels'][i]] == 'person':
        person_masks.append(masks[i].detach().numpy())
proba_threshold = 0.1
person_masks = np.array(person_masks)
bool_masks = (person_masks > proba_threshold)

""" Erosion and dilation 
kernel = np.ones((30,30), np.uint)
bool_masks = (bool_masks.astype('uint8')*255).astype('uint8')
bool_masks = cv2.erode(bool_masks, kernel, iterations = 1)
bool_masks = cv2.dilate(bool_masks, kernel, iterations = 1)
bool_masks = cv2.erode(bool_masks, kernel, iterations = 1)
bool_masks = cv2.dilate(bool_masks, kernel, iterations = 1)
bool_masks = bool_masks / 225
"""
K = as_intrinsics_matrix([fx, fy, cx, cy])

""" Homography estimation """

_, prev_color, prev_depth = frame_reader("./test_img/1548339828.01926.png", "./test_img/1548339827.96356.png")
cv2.imwrite('prev_rgb.png', prev_color.numpy()*255)
cv2.imwrite('prev_depth.png', prev_depth.numpy()*png_depth_scale)

prev_color_img = prev_color.detach().cpu().numpy()
prev_color_img = prev_color_img * 255.
prev_color_img = prev_color_img.astype(np.uint8)

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(prev_color_img,None)
kp2, des2 = orb.detectAndCompute(curr_color_img,None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptor
matches = bf.match(des1,des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)          
refPts = np.array([kp1[m.queryIdx].pt for m in matches])
currPts = np.array([kp2[m.trainIdx].pt for m in matches])

# Compute the homography from camera (k+1) to camera (k)
H, _ = cv2.findHomography(currPts[:10,:], refPts[:10,:], cv2.RANSAC, 0.5)

""" Pose estimation """
# Estimate the essential matrix
E, _ = cv2.findEssentialMat(currPts, refPts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Decompose the pose of camera (k+1) relative to camera (k)
_ , neighbor_R, neighbor_t, _ = cv2.recoverPose(E, currPts, refPts, K)
neighbor_t = neighbor_t * rescale_translation_factor(refPts, currPts)

prev_color_img = np.transpose(prev_color_img, (1,0,2))  # [h, w, c] -> [w, h ,c]
curr_color_img = np.transpose(curr_color_img, (1,0,2)) 
prev_depth_img = np.transpose(prev_depth, (1,0))
curr_depth_img = np.transpose(gt_depth, (1,0))

""" Background inpainting pixel by pixel """
for i in range(bool_masks.shape[0]):
    for h in range(bool_masks.shape[1]):
        for w in range(bool_masks.shape[2]):
            if bool_masks[i][h][w] == True:
                print("Processing", h, w)
                color_cnt = 0
                depth_cnt = 0
                sum_color_r = 0
                sum_color_g = 0
                sum_color_b = 0
                sum_depth = 0

                scale_factor = 1 / (H[2,0]*w + H[2,1]*h + H[2,2])
                back_proj = scale_factor * np.matmul(H, np.array([w, h, 1]))
                
                # Bilinear interpolate
                x1, y1 = math.floor(back_proj[0]), math.floor(back_proj[1])
                x2, y2 = math.ceil(back_proj[0]), math.floor(back_proj[1])
                x3, y3 = math.ceil(back_proj[0]), math.ceil(back_proj[1])
                x4, y4 = math.floor(back_proj[0]), math.ceil(back_proj[1])

                try:
                    if prev_depth_img[x1, y1] >= 0 and prev_depth_img[x2, y2] >= 0 and prev_depth_img[x3, y3] >= 0 and prev_depth_img[x4, y4] >= 0:
                        w1 = (x3 - back_proj[0])*(y3 - back_proj[1])
                        w2 = (back_proj[0] - x4)*(y4 - back_proj[1])
                        w3 = (back_proj[0] - x1)*(back_proj[1] - y1)
                        w4 = (x2 - back_proj[0])*(back_proj[1] - y2)
                        sum_color_r +=  w1*prev_color_img[x1, y1, 0] + w2*prev_color_img[x2, y2, 0] + w3*prev_color_img[x3, y3, 0] + w4*prev_color_img[x4, y4, 0]
                        sum_color_g +=  w1*prev_color_img[x1, y1, 1] + w2*prev_color_img[x2, y2, 1] + w3*prev_color_img[x3, y3, 1] + w4*prev_color_img[x4, y4, 1]
                        sum_color_b +=  w1*prev_color_img[x1, y1, 2] + w2*prev_color_img[x2, y2, 2] + w3*prev_color_img[x3, y3, 2] + w4*prev_color_img[x4, y4, 2]
                        
                        color_cnt += 1
                        # projMat = np.concatenate((np.concatenate((neighbor_R, neighbor_t), axis=1), [[0.,0.,0.,1.]]), axis=0)
                        # depth_vector = projMat @ np.array([[0], [0], [prev_depth_img[w][h]], [1]])
                        # Triangulation
                        ix = int(back_proj[0]) if (back_proj[0] - int(back_proj[0]) < 0.5) else (int(back_proj[0])+1)
                        iy = int(back_proj[1]) if (back_proj[1] - int(back_proj[1]) < 0.5) else (int(back_proj[1])+1)
                        
                        if prev_depth_img[ix][iy] != 0:
                            currProj = K @ np.eye(3, 4, dtype=np.float64) # curr -> prev
                            prevProj = K @ np.concatenate((neighbor_R, neighbor_t), axis=1)
                            
                            prevPt = np.array([[back_proj[0]], [back_proj[1]]])
                            currPt = np.array([[w], [h]])
                            
                            scenePt = cv2.triangulatePoints(prevProj, currProj, prevPt, currPt)
                            scenePt = cv2.convertPointsFromHomogeneous(scenePt.T).squeeze()
                            
                            p2w_tmp = np.linalg.inv(K) @ np.array([[w], [h], [1]])
                            p2w_tmp = p2w_tmp.reshape(3)
                            
                            prev_p2w_tmp = np.linalg.inv(K) @ np.array([[back_proj[0]], [back_proj[1]], [1]])
                            prev_p2w_tmp = np.linalg.inv(neighbor_R) @ (prev_p2w_tmp - neighbor_t)
                            prev_p2w_tmp = prev_p2w_tmp.reshape(3)
                            
                            px, py, pz = prev_p2w_tmp[0], prev_p2w_tmp[1], prev_p2w_tmp[2]
                            sx, sy, sz = scenePt[0], scenePt[1], scenePt[2]
                            depth_scale =  prev_depth_img[ix][iy] / np.linalg.norm(prev_p2w_tmp - scenePt)

                            sum_depth += (depth_scale * np.linalg.norm(p2w_tmp - scenePt))
                            depth_cnt += 1
                except: print("exception")     
                if color_cnt!=0: # Handle division by zero exception
                    print("change color")
                    curr_color_img[w][h][0] = sum_color_r / color_cnt
                    curr_color_img[w][h][1] = sum_color_g / color_cnt
                    curr_color_img[w][h][2] = sum_color_b / color_cnt
                
                if depth_cnt!=0:
                    curr_depth_img[w][h] = sum_depth / depth_cnt
                else:
                    curr_depth_img[w][h] = 0

gt_color = torch.from_numpy(curr_color_img / 255.).permute(1,0,2)
gt_depth = curr_depth_img.permute(1,0)
cv2.imwrite('inpainted_rgb.png', gt_color.numpy()*255)
cv2.imwrite('inpainted_depth.png', gt_depth.numpy()*png_depth_scale)
