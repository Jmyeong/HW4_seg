import numpy as np
import cv2
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

def dice_score_binary(pred_mask: np.ndarray, true_mask: np.ndarray, smooth: float = 1e-6):
    # 넘파이 배열을 PyTorch 텐서로 변환
    pred_mask = torch.tensor(pred_mask, dtype=torch.float32) / 255.0
    true_mask = torch.tensor(true_mask, dtype=torch.float32) / 255.0
    # print(torch.unique(pred_mask))
    # print(torch.unique(true_mask))
    # 교집합과 합집합 계산
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum()

    # Dice score 계산
    dice = (2. * intersection + smooth) / (union + smooth)
    # print(dice)
    return dice.item()

def compare_image(gt_dict, test_dict):
    for (gt_key, gt), (test_key, test) in zip(gt_dict.items(), test_dict.items()):
        # print(key)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(gt)
        plt.title(gt_key.split("/")[-1])
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(test)
        plt.title(test_key.split("/")[-1])
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
        
def make_dict(image_list, image_name_list):
    new_dict = {}
    for image, name in zip(image_list, image_name_list):
        new_dict[name] = image
    return new_dict

test_path = "./outputs_94"
test_lists = glob(test_path + "/**/*.tif", recursive=True)
test_image_list = []
for test in test_lists:
    image = Image.open(test)
    image = np.array(image)
    test_image_list.append(image)
        
gt_path = "./HW4_data/kaggle_3m"
gt_lists = glob(gt_path + "/**/*.tif", recursive=True)
gt_image_list = []
final_test_image_list = []
final_test_list = []
gt_name_list = []

for gt in gt_lists:
    for test in test_lists:
        if test.split("/")[-1] == gt.split("/")[-1]:
            # print(test)
            # print(gt)
            gt_image = Image.open(gt)
            gt_image = np.array(gt_image)
            test_image = Image.open(test)
            test_image = np.array(test_image)
            gt_image_list.append(gt_image)
            gt_name_list.append(gt)
            final_test_image_list.append(test_image)
            final_test_list.append(test)
            
gt_name_list = sorted(gt_name_list)
final_test_list = sorted(final_test_list)
     
gt_dict = make_dict(image_list=gt_image_list, image_name_list=gt_name_list)
test_dict = make_dict(image_list=final_test_image_list, image_name_list=final_test_list)

# print(len(gt_dict))

dice_scores = []

for i in range(len(gt_image_list)):
    Dice = dice_score_binary(gt_image_list[i], final_test_image_list[i])
    # print(Dice)
    dice_scores.append(Dice)

# compare_image(gt_dict=gt_dict, test_dict=test_dict)

avg_dice = np.mean(dice_scores)
print(f"Avg Dice score : {avg_dice}")