'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.
    
    ### Extract key points and their respective features for each image ###
    # Define saving dictionaries
    keypoints = {}
    features = {}
    # Iterate through images
    for img_num, img_array in imgs.items():
        # Convert to float to match SIFT function expected input
        img_array = img_array.float()
        # Add channel to match SIFT function expected input
        if len(img_array.shape) == 3:
            img_array = img_array.unsqueeze(0)
        # Convert to grayscale to match SIFT function expected input
        if img_array.shape[1] == 3:
            img_array_g = K.color.rgb_to_grayscale(img_array)
        else:
            img_array_g = img_array
        # Use SIFT to extract key points and features
        loc_affine_frms, resp_func_vals, loc_descs = K.feature.SIFTFeature()(img_array_g)
        # Save key points and features
        keypoints[img_num] = loc_affine_frms
        features[img_num] = loc_descs
    
    ### Match features ###
    # Extract features per image
    img_num = list(features.keys())
    feat1 = features[img_num[0]]
    feat2 = features[img_num[1]]
    # Batch compute SSD
    ssd = torch.cdist(feat1, feat2, p=2.0) ** 2
    # Find best (f1-f2) and 2nd best (f1-f2') match for each feature
    distances, indices = torch.topk(ssd, k=2, largest=False)
    best_ssd = distances[..., 0]
    second_best_ssd = distances[..., 1]
    # Calculate ssd ratio distances
    ratio_distances = best_ssd / second_best_ssd
    # Filter valid matches using arbitrary threshold
    threshold = 0.6
    valid_matches = ratio_distances < threshold
    
    ### Use matches to determine overlap between images ###
    # Arbitrary match threshold to determine overlap
    overlap_threshold = 15 # has to be at least 8 for projection matrix d.o.f
    overlap = valid_matches.sum().item() >= overlap_threshold
    if overlap:
        # Extract keypoints for each image
        keypoints1_batched = keypoints[img_num[0]][..., :, 2]
        keypoints2_batched = keypoints[img_num[1]][..., :, 2]
        keypoints1 = keypoints1_batched[0]
        keypoints2 = keypoints2_batched[0]
        # Define indices for matched points in image 1
        indices1 = torch.where(valid_matches[0])[0]
        # Use indices from topk for matched points in image 2
        indices2 = indices[0, valid_matches[0],0]
        print(indices1.shape)
        print(indices2.shape)
    else:
        print("No Overlap")
        
    return img

# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    img = torch.zeros((3, 256, 256)) # assumed 256*256 resolution. Update this as per your logic.
    overlap = torch.empty((3, 256, 256)) # assumed empty 256*256 overlap. Update this as per your logic.

    #TODO: Add your code here. Do not modify the return and input arguments.

    return img, overlap
