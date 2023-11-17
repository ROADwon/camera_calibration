import cv2 as cv
import numpy as np
import glob
import os
import sys
import torch
import torchvision
import time

from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

img_path = "C:/Users/line/Desktop/stitch and perspective/imgs"
img_save_path = "C:/Users/line/Desktop/maskRCNN/imgs" 
img_list = [cv.imread(file) for file in glob.glob(f'{img_path}/*jpg')]
images = []

model_type = "vit_h"
model_checkpoint = "sam_vit_h_4b8939.pth"

device = "cuda"

if torch.cuda.is_available :
    torch.device("cuda")
else : 
    torch.device("cpu")
    

for image in img_list :
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
    image = cv.resize(image,(1080,1440))
    images.append(image)


def show_mask(mask,ax, random_color = False):
    if random_color :
        color = np.concatenate([np.random.random(3), np.array([0,6])], axis= 0)
    else :
        color = np.array([1/255, 1/255, 1/255, .9])  #RGBA(적,녹,청,투명)
    h, w= mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size = 100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    ax.scatter(pos_points[:, 0], pos_points[:,1], color="green", marker="*", s=marker_size, edgecolor="black", linewidth=.3)
    ax.scatter(neg_points[:, 0], neg_points[:,1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=.3)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[1], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0,0,0,0), lw=2))
    
sam = sam_model_registry[model_type](checkpoint=model_checkpoint)
sam.to(device=device)
x_point, y_point = images[0].shape[1] //2, images[0].shape[0]//2

input_point = np.array([[x_point,y_point]])
input_label = np.array([1])

predictor = SamPredictor(sam)
predict_images = []
black_white_images=[]
for i in range(len(images)):
    start = time.time()
    predict_image = predictor.set_image(images[i])
    predict_images.append(predict_image)
    
    masks, scores, logits = predictor.predict(
        point_coords= input_point,
        point_labels= input_label,
        multimask_output= True
    )
    max_index = np.argmax(scores)
    img = masks[max_index].astype(np.uint8) *255
    #img = masks[1].astype(np.uint8) *255
    black_white_images.append(img)
    cv.imwrite(f'{img_save_path}/img{i}.jpg',black_white_images[i])
    end = time.time()
    print(f"=========== finished {i+1}images ===========")
    print(f"===== remaining {len(images)- i} images =====")
    print(f"=== Time required : {end - start:.5f}sec ===")
    
    