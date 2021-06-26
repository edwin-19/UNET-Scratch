# UNET Tensorflow 2 
- This repo is a sample for writing UNET from scratch to have an understanding on the architecture

# Metrics
The metric used here to evluate segmentation is pixel wise IOU and mIoU

You can view the wonderful tutorial here: https://www.jeremyjordan.me/evaluating-image-segmentation-models/
```python
# Note this function is available in utils as well
import numpy as np

def iou(gt_mask, pred_mask):
    intersection = np.logical_and(gt_mask, pred_mask)
    union = np.logical_or(gt_mask, pred_mask)
    
    return np.sum(intersection) / np.sum(union)
```

# Example Inference
To run the inference here's an example
```python
python demo.jpg --save
```

![example](results/demo.jpg)

# TODO
- [x] Mini Unet
- [x] Evaluation Script
- [x] Inference Script
- [x] UNET
- [ ] UNET Backbone

# Model Comparison
| Model Name | Class 1 | Class 2 | Class 3 |
|------------|---------|---------|---------|
| Mini UNET  | 34.17   | 58.18   | 7.38    |
| UNET Plain | 73.63   | 82.83   | 39.4    |