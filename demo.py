import tensorflow as tf
import argparse
from matplotlib import pyplot as plt
import numpy as np
import os

def preprocess(image_path):
    img = tf.io.decode_jpeg(tf.io.read_file(image_path))
    img_norm = tf.cast(img, tf.float32)
    img_norm = tf.image.resize(img_norm, [256, 256]) / 255.
    img_norm = tf.expand_dims(img_norm, axis=0)
    return img, img_norm
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d','--data', type=str, default='data/test/test.jpg'
    )
    
    parser.add_argument(
        '-m', '--model', type=str, default='model/model.h5'
    )
    
    parser.add_argument(
        '-s', '--show', action='store_false'
    )
    
    parser.add_argument(
        '--save', action='store_false'
    )
    args = parser.parse_args()
    
    model = tf.keras.models.load_model(args.model, compile=False)
    img, img_norm = preprocess(args.data)
    
    mask_pred = model.predict(img_norm)
    mask_pred = np.squeeze(mask_pred, axis=0)
    mask_pred = (mask_pred * 255).astype(np.uint8)
    
    img = tf.image.resize(img, [256, 256])
    img = tf.cast(img, tf.uint8)
    
    preds_gt = np.hstack([img, mask_pred])
    
    if args.save:
        if not os.path.exists('results/'):
            os.makedirs('results/')
            
        tf.keras.preprocessing.image.save_img(
            'results/demo.jpg', preds_gt
        )
    
    if args.show:
        plt.imshow(preds_gt)
        plt.show()