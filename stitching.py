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
        # Use Kornia's SIFT function to extract key points and features
        torch.manual_seed(42) # Prevents different output every time
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
        # Extract matches points per image
        matched_points1 = keypoints1[indices1]
        matched_points2 = keypoints2[indices2]

        ### Compute homography between pairs using RANSAC ###
        
        # Use Kornia's RANSAC function to compute homography
        homography, _ = K.geometry.ransac.RANSAC(model_type='homography')(matched_points1, matched_points2)
        
        ### Transform the images and stitch them into one mosaic, eliminating the foreground ###
        
        # Step 1. Calculate mosaic canvas size
        # Extract each image and convert to float
        # img1 = imgs[img_num[0]]
        # img2 = imgs[img_num[1]]
        # show_image(img1)
        # show_image(img2)
        img1 = imgs[img_num[0]].float()
        img2 = imgs[img_num[1]].float()
        # Extract image dimensions
        c, h1, w1 = img1.shape
        _, h2, w2 = img2.shape
        # Define pre-warp img1 corners and convert to homogenous coordinates
        corners1 = torch.tensor([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        h_corners1 = torch.cat([corners1, torch.ones(4, 1)], dim=1).unsqueeze(-1)
        # Warp img1 corners using homography matrix and convert back to img coords
        warp_h_corners1 = torch.matmul(homography, h_corners1).squeeze(-1)
        warp_corners1 = warp_h_corners1[:, :2] / warp_h_corners1[:, 2:]
        # Determine min, max coordinates from warped img1 and original img2
        corners2 = torch.tensor([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        all_corners = torch.cat([warp_corners1, corners2])
        x_min, y_min = torch.floor(all_corners.min(dim=0)[0]).int()
        x_max, y_max = torch.ceil(all_corners.max(dim=0)[0]).int()
        canvas_w = x_max - x_min
        canvas_h = y_max - y_min
        
        # Step 2. Calculate translation and perspective matrices for images to canvas
        # Create translation matrix to shift all img coords to be positive
        translation = torch.eye(3)
        translation[0, 2] = -x_min
        translation[1, 2] = -y_min
        # Multiply translation matrix by homography matrix to make perspective transformation matrix
        perspective = torch.matmul(translation, homography).unsqueeze(0)
        
        # Step 3. Warp images to canvas
        # Use Kornia's warp perspective function on both images (apply only translation to img1)
        warp_img1 = K.geometry.warp_perspective(img1.unsqueeze(0), perspective, (canvas_h, canvas_w))[0]
        warp_img2 = K.geometry.warp_perspective(img2.unsqueeze(0), translation.unsqueeze(0), (canvas_h, canvas_w))[0]
        
        # Step 4. Set up foreground elimination strategy
        # Define mask where pixels exist from either image
        mask1 = (warp_img1.sum(dim=0) > 0).float()
        mask2 = (warp_img2.sum(dim=0) > 0).float()
        overlap_mask = mask1 * mask2
        # Calculate norm diff between imgs in overlap mask and threshold for foreground
        diff = torch.norm((warp_img1 - warp_img2) / 255.0, dim=0)
        foreground_mask = (diff > 0.2) * overlap_mask        
        # Determine which image contains the background at foreground mask pixels
        # Use non-overlapping regions as guaranteed background
        only_mask1 = (mask1 * (1 - mask2))
        only_mask2 = (mask2 * (1 - mask1))
        # Calculate number of pixels for average calculation
        n1 = only_mask1.sum().clamp(min=1.0)
        n2 = only_mask2.sum().clamp(min=1.0)
        # Derive mean background color per image from its non-overlapping region
        bg1 = (warp_img1 * only_mask1.unsqueeze(0)).sum(dim=(1, 2), keepdim=True) / n1
        bg2 = (warp_img2 * only_mask2.unsqueeze(0)).sum(dim=(1, 2), keepdim=True) / n2
        # Per-pixel distance from each image's own background model
        dist1 = torch.norm((warp_img1 - bg1) / 255, dim=0)
        dist2 = torch.norm((warp_img2 - bg2) / 255, dim=0)
         
        # Step 5. Construct mosaic
        # Initialize zero matrix as canvas for mosaic
        img = torch.zeros(c, canvas_h, canvas_w)
        # Add imgs to canvas
        img = torch.where(mask1.bool().unsqueeze(0), warp_img1, img)
        img = torch.where(mask2.bool().unsqueeze(0), warp_img2, img)
        # Pixels in overlap mask are applied blending strategy
        # Non-foreground overlap pixels take average values from both images
        average_overlap = (warp_img1 + warp_img2) / 2.0
        # Foreground mask pixels take values from image with smaller dist from median
        # Create array which represents the image with smaller distance from median at each pixel
        dist = (dist1 <= dist2).float().unsqueeze(0)
        smaller_dist = dist * warp_img1 + (1 - dist) * warp_img2
        # Apply blending strategy to relevant masked pixels
        blended = torch.where(foreground_mask.bool().unsqueeze(0), smaller_dist, average_overlap)
        img = torch.where(overlap_mask.bool().unsqueeze(0), blended, img)
        
        ### Save resulting mosaic ###
        # Convert mosaic to uint8
        img = img.to(torch.uint8)
        # show_image(img)
        
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
    
    # Initialize overlap array
    img_nums = list(imgs.keys())
    N = len(img_nums)
    overlap = torch.eye(N)
    
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
        # Use Kornia's SIFT function to extract key points and features
        torch.manual_seed(42) # Prevents different output every time
        loc_affine_frms, resp_func_vals, loc_descs = K.feature.SIFTFeature()(img_array_g)
        # Save key points and features
        keypoints[img_num] = loc_affine_frms
        features[img_num] = loc_descs
        
    ### Match features and build overlap array ###
    
    # Loop through image pairs
    homographies = {}
    for i in range(N):
        for j in range(N):
            # Homography between image and itself should be identity matrix
            if i == j:
                homographies[(i, j)] = torch.eye(3)
                continue
            # Extract features per image
            feat1 = features[img_nums[i]]
            feat2 = features[img_nums[j]]
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
            # Arbitrary match threshold to determine feature overlap
            feat_overlap_threshold = 15 # has to be at least 8 for projection matrix d.o.f
            feat_overlap = valid_matches.sum().item() >= feat_overlap_threshold
            if feat_overlap:
                # Extract keypoints for each image
                keypoints1_batched = keypoints[img_nums[i]][..., :, 2]
                keypoints2_batched = keypoints[img_nums[j]][..., :, 2]
                keypoints1 = keypoints1_batched[0]
                keypoints2 = keypoints2_batched[0]
                # Define indices for matched points in image 1
                indices1 = torch.where(valid_matches[0])[0]
                # Use indices from topk for matched points in image 2
                indices2 = indices[0, valid_matches[0],0]
                # Extract matches points per image
                matched_points1 = keypoints1[indices1]
                matched_points2 = keypoints2[indices2]
                # Use Kornia's RANSAC function to compute homography
                homography, _ = K.geometry.ransac.RANSAC(model_type='homography')(matched_points1, matched_points2)
                
                ### Determine if the images spatially overlap by 20% ###
                
                # Extract dimensions of images
                c_i, h_i, w_i = imgs[img_nums[i]].shape
                c_j, h_j, w_j = imgs[img_nums[j]].shape
                # Make dummy ones matrix that represents image i
                dummy_i = torch.ones(1, 1, h_i, w_i)
                # Project dummy i onto image j canvas, removing any dummy i  not overlapping
                warp_dummy_i = K.geometry.warp_perspective(dummy_i, homography.unsqueeze(0), (h_j, w_j))[0, 0]
                # Count number of remaining pixels (0.5 threshold used due to bilinear interpolation)
                overlap_pixels = (warp_dummy_i > 0.5).sum()
                # Calculate overlap percentage
                area_i = h_i * w_i
                area_j = h_j * w_j
                min_area = min(area_i, area_j)
                overlap_pct = overlap_pixels / min_area
                if overlap_pct >= 0.2:
                    homographies[(i, j)] = homography
                    overlap[i, j] = 1.0
    
    ### Calculate global homography links to anchor image using breadth-first search ###
    
    # Define anchor as the image with the most overlaps
    num_overlaps = overlap.sum(dim=1)
    anchor_idx = torch.argmax(num_overlaps).item()
    # Initialize global homographies dictionary (each image has a homography linking back to anchor)
    homography_glob = {anchor_idx: torch.eye(3)}
    # Set up queue to check for overlapping images with breadth-first search
    visited = []
    queue = [anchor_idx]
    while queue:
        current = queue.pop(0)
        for new in range(N):
            if new not in visited and overlap[current, new] == 1.0:
                visited.append(new)
                queue.append(new)
                # Map new image to reference (check for reliable homography)
                if (new, current) in homographies:
                    new_to_current = homographies[(new, current)]
                elif (current, new) in homographies:
                    new_to_current = homographies[(current, new)]
                else:
                    new_to_current = torch.eye(3)
                homography_glob[new] = torch.matmul(homography_glob[current], new_to_current)
    
    ### Build canvas for panorama ###
    
    # Initialize min and max coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    # Get all connected images that map to anchor canvas
    valid_indices = list(homography_glob.keys())
    # Calculate global bounding box for panorama
    for idx in valid_indices:
        img_array = imgs[img_nums[idx]].float()
        c, h, w = img_array.shape
        # Define img corners and convert to homogenous coordinates
        corners = torch.tensor([[0, 0], [w, 0], [w, h], [0, h]])
        h_corners = torch.cat([corners, torch.ones(4, 1)], dim=1).unsqueeze(-1)
        # Warp corners using global homography matrix
        warp_h_corners = torch.matmul(homography_glob[idx], h_corners).squeeze(-1)
        warp_corners = warp_h_corners[:, :2] / warp_h_corners[:, 2:]
        # Update min and max coordinates
        min_x = min(min_x, torch.floor(warp_corners[:, 0].min()))
        min_y = min(min_y, torch.floor(warp_corners[:, 1].min()))
        max_x = max(max_x, torch.ceil(warp_corners[:, 0].max()))
        max_y = max(max_y, torch.ceil(warp_corners[:, 1].max()))
    
    return img, overlap
