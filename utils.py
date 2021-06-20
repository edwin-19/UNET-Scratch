from matplotlib import pyplot as plt

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.title(name + "_" + label.lower())
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        
    plt.show()