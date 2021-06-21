from matplotlib import pyplot as plt
import numpy as np

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.title(name)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        
    plt.show()
    
def iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    
    return np.sum(intersection) / np.sum(union)