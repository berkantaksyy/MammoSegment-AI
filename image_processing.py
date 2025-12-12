import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score, f1_score
import math

# Function to get accuracy, dice and jaccard scores
def get_scores(y_true, y_pred):
    # flatten arrays just in case
    yt = y_true.flatten()
    yp = y_pred.flatten()
    
    dice = f1_score(yt, yp, average='binary')
    iou = jaccard_score(yt, yp, average='binary')
    acc = np.mean(yt == yp)
    
    return acc, dice, iou

# 1. Load Data
# File paths
f_img = '00b2567654cfadcda25f5bcbe44f4974_breast.jpg'
f_mask = '00b2567654cfadcda25f5bcbe44f4974_mask.jpg'

# Load as grayscale
img = cv2.imread(f_img, 0)
gt_img = cv2.imread(f_mask, 0)

# Binarize ground truth (0 or 1)
ret, gt_bin = cv2.threshold(gt_img, 127, 1, cv2.THRESH_BINARY)
h, w = img.shape

# 2. Preprocessing
# Using CLAHE for better contrast
cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
im_cl = cl.apply(img)
# Median blur to remove salt-and-pepper noise
im_blur = cv2.medianBlur(im_cl, 5)

# Plot Preprocessing
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(im_blur, cmap='gray'); plt.title("CLAHE + Blur")
plt.axis('off')
plt.savefig("Fig1_Process.png", bbox_inches='tight')
plt.show()

# 3. Segmentation (Otsu)
# Get initial mask
_, th = cv2.threshold(im_blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Morphological ops to close gaps
k = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)

# Copy for later visualization
mask_step1 = mask.copy() * 255 

# 4. Remove Pectoral Muscle
# Copy mask to modify it
final_res = mask.copy()
viz_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # for debug lines

# Estimate muscle area based on top row pixels
row_chk = mask[5, :] 
whites = np.where(row_chk > 0)[0]

# Simple logic to find cutting point
if len(whites) > 0:
    x_lim = whites[-1]
else:
    x_lim = int(w * 0.3) # default fallback

y_lim = int(x_lim * 2.5)
if y_lim > h * 0.6: 
    y_lim = int(h * 0.6)

# Edge detection on the specific ROI
roi_mask = np.zeros_like(mask)
cv2.rectangle(roi_mask, (0,0), (x_lim + 100, y_lim + 100), 255, -1)

# Mask the blurred image
masked_im = cv2.bitwise_and(im_blur, im_blur, mask=mask)
masked_im = cv2.bitwise_and(masked_im, masked_im, mask=roi_mask)

# Canny edges
edges = cv2.Canny(masked_im, 30, 100)

# Probabilistic Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=15, minLineLength=30, maxLineGap=20)

cut_done = False
best_ln = None

if lines is not None:
    max_len = 0
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        # Calculate angle in degrees
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Filter angles (muscle is usually diagonal)
        if -85 < ang < -20:
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if dist > max_len:
                max_len = dist
                best_ln = (x1, y1, x2, y2)
    
    # If we found a good line, cut the mask
    if best_ln:
        x1, y1, x2, y2 = best_ln
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            c = y1 - m * x1
            
            # Create polygon to make black
            poly = np.array([[0, 0], [w, 0], [w, int(m*w + c)], [0, int(c)]])
            cv2.fillPoly(final_res, [poly], 0)
            
            # Draw green line for report
            cv2.line(viz_img, (x1, y1), (x2, y2), (0, 255, 0), 3) 
            cut_done = True

# Fallback: if hough failed, cut geometrically
if not cut_done:
    pts = np.array([[0,0], [x_lim, 0], [0, y_lim]])
    cv2.fillPoly(final_res, [pts], 0)
    cv2.line(viz_img, (x_lim, 0), (0, y_lim), (255, 0, 0), 3) # Blue line

# Plot Method details
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(edges, cmap='gray'); plt.title("Canny Edges")
plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(viz_img); plt.title("Detected Boundary")
plt.axis('off')
plt.savefig("Fig2_Method.png", bbox_inches='tight')
plt.show()

mask_step2 = final_res.copy() * 255 

# 5. Post-Processing / Cleaning
# Remove artifacts at the bottom
bottom_thresh = 2600
if h > bottom_thresh: 
    final_res[bottom_thresh:, :] = 0

# Keep only the largest object (removes nipple/noise)
n_labels, labels = cv2.connectedComponents(final_res)

if n_labels > 1:
    # get areas (ignoring background)
    areas = np.bincount(labels.flatten())[1:]
    big_idx = np.argmax(areas) + 1
    
    # create clean mask
    temp = np.zeros_like(final_res)
    temp[labels == big_idx] = 255
    final_res = temp

mask_step3 = final_res.copy()

# Plot Evolution steps
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(mask_step1, cmap='gray'); plt.title("1. Otsu")
plt.axis('off')
plt.subplot(1, 3, 2); plt.imshow(mask_step2, cmap='gray'); plt.title("2. Muscle Cut")
plt.axis('off')
plt.subplot(1, 3, 3); plt.imshow(mask_step3, cmap='gray'); plt.title("3. Final Clean")
plt.axis('off')
plt.savefig("Fig3_Steps.png", bbox_inches='tight')
plt.show()

# 6. Results & Metrics
# normalize to 0-1 for scoring
res_bin = final_res // 255
acc, d_score, j_score = get_scores(gt_bin, res_bin)

print("-" * 30)
print("FINAL SCORES")
print("-" * 30)
print(f"Dice Score:      {d_score:.4f}")
print(f"Jaccard (IoU):   {j_score:.4f}")
print(f"Pixel Accuracy:  {acc:.4f}")
print("-" * 30)

# Show errors
diff = cv2.absdiff(gt_bin * 255, final_res)

# Final comparison plot
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1); plt.imshow(img, cmap='gray'); plt.title("Original")
plt.axis('off')
plt.subplot(2, 2, 2); plt.imshow(gt_img, cmap='gray'); plt.title("Ground Truth")
plt.axis('off')
plt.subplot(2, 2, 3); plt.imshow(final_res, cmap='gray'); plt.title(f"Our Result (Dice: {d_score:.3f})")
plt.axis('off')
plt.subplot(2, 2, 4); plt.imshow(diff, cmap='hot'); plt.title("Error Map")
plt.axis('off')

plt.tight_layout()
plt.savefig("Fig4_Result.png", bbox_inches='tight')
plt.show()