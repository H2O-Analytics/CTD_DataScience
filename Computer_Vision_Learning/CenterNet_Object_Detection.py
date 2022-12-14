import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import tensorflow_hub as hub
import urllib
# Check for TensorFlow GPU access
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# See TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Code location: https://learnopencv.com/centernet-anchor-free-object-detection-explained/

models = {
          'Resnet101':'https://tfhub.dev/tensorflow/centernet/resnet101v1_fpn_512x512/1',   
          'HourGlass104':'https://tfhub.dev/tensorflow/centernet/hourglass_512x512/1'
         }

images = [
          'https://farm7.staticflickr.com/6073/6032446158_85fa667cd2_z.jpg',
          'https://farm9.staticflickr.com/8538/8678472399_886f8eabec_z.jpg',
          'https://farm6.staticflickr.com/5485/10028794463_d8cbb38932_z.jpg',
          'https://farm4.staticflickr.com/3057/2475401198_0a342a907e_z.jpg'
         ]

# Download images.
for i in range(len(images)):
    urllib.request.urlretrieve(images[i], "img{}.jpg".format(i+1))

# Read Images.
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')
img3 = cv2.imread('img3.jpg')
img4 = cv2.imread('img4.jpg')

# Plot Images
def plot_images(img_list, title=None, row=1, column=2, 
                fig_size=(10, 15)):
    plt.figure(figsize=fig_size)
    for i, img in enumerate(img_list):
        plt.subplot(row, column, i+1)
        plt.imshow(img[...,::-1])
        plt.axis('off')
        plt.title(title[i] if title else 'img{}'.format(i+1))
    plt.show()

image_list = [img1, img2, img3, img4]
plot_images(image_list, row=2, column=2, fig_size=(15, 10))

# Define class names for the COCO dataset. (Tensorflow models are trained on the COCO dataset)
category_index = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 
                  5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 
                  10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 
                  14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 
                  18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 
                  22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 
                  27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 
                  33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 
                  37: 'sports ball', 38: 'kite', 39: 'baseball bat', 
                  40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 
                  43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 
                  47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl',  
                  52: 'banana',  53: 'apple',  54: 'sandwich',  55: 'orange',  
                  56: 'broccoli',  57: 'carrot',  58: 'hot dog',  59: 'pizza',  
                  60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 
                  64: 'potted plant', 65: 'bed', 67: 'dining table', 
                  70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 
                  75: 'remote', 76: 'keyboard', 77: 'cell phone', 
                  78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 
                  82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 
                  87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 
                  90: 'toothbrush'}


# Class IDs to Class colors
R = np.array(np.arange(0,256,63))
G = np.roll(R,2)
B = np.roll(R,4)

COLOR_IDS = np.array(np.meshgrid(R,G,B)).T.reshape(-1,3)

# Load model from TF Hub
# ResNet101.
resnet = hub.load(models['Resnet101'])
# Hourglass104.
hourglass = hub.load(models['HourGlass104'])


# Run Inference
# An inference outcome is a dictionary with the following keys:
    # Number of detections
    # Detection boxes
    # Detection scores
    # Detection classes

# Hourglass inference
result_hourglass = hourglass(np.array([img1]))
result_hourglass.keys()

# Check shape of predictions
# total number of bounding boxes = 100
print('num_detections shape\t:{}'.format(result_hourglass['num_detections'].shape))
print('detection_boxes shape\t:{}'.format(result_hourglass['detection_boxes'].shape))
print('detection_scores shape\t:{}'.format(result_hourglass['detection_scores'].shape))
print('detection_classes shape\t:{}'.format(result_hourglass['detection_classes'].shape))

# Convert tensors to numpy
def to_numpy(prediction):
    result = dict()
    bboxes = prediction['detection_boxes'][0].numpy()
    scores = prediction['detection_scores'][0].numpy()
    # class ids are int
    classes = prediction['detection_classes'][0].numpy().astype(int)
    return bboxes, scores, classes

print_count = 5
bboxes, scores, classes = to_numpy(result_hourglass)

# print results
print('detection_boxes:\n{}'.format(bboxes[:print_count]))
print('detection_scores:\n{}'.format(scores[:print_count]))
print('detection_classes:\n{}'.format(classes[:print_count]))

# Filter confident predictions
def filter_detections_on_score(boxes, scores, classes, score_thresh=0.3):
    ids = np.where(scores >= score_thresh)
    return boxes[ids], scores[ids], classes[ids]

score_thresh = 0.30
bboxes, scores, classes = filter_detections_on_score(bboxes, scores, classes, 
                                                    score_thresh)
print('detection_boxes:\n{}'.format(bboxes))
print('detection_scores:\n{}'.format(scores))
print('detection_classes:\n{}'.format(classes))

# Convert normalized output to pixels
def normalize_to_pixels_bboxs(bboxes, img):
    img_height, img_width, _ = img.shape
    bboxes[:, 0] *= img_height
    bboxes[:, 1] *= img_width
    bboxes[:, 2] *= img_height
    bboxes[:, 3] *= img_width
    return bboxes.astype(int)

bboxes = normalize_to_pixels_bboxs(bboxes, img1)
print('detection_boxes:\n{}'.format(bboxes))

# Annotate detection boxes
def add_prediction_to_image(img, bboxes, scores, classes, id_class_map=category_index, colors=COLOR_IDS):
    img_with_bbox = img.copy()
    for box, score, cls in zip(bboxes, scores, classes):
        top, left, bottom, right = box
        class_name = id_class_map[cls]
 
        # Bounding box annotations.
        color = tuple(colors[cls % len(COLOR_IDS)].tolist())[::-1]
        img_with_bbox = cv2.rectangle(img_with_bbox, (left, top), (right, bottom), color, thickness=2)
        display_txt = '{}: {:.2f}'.format(class_name, score)
        ((text_width, text_height), _) = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2) 
        img_with_bbox = cv2.rectangle(img_with_bbox, (left, top - int(0.9 * text_height)), (left + int(0.4*text_width), top), color, thickness=-1)
        img_with_bbox = cv2.putText(img_with_bbox, display_txt, (left, top - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
 
    return img_with_bbox

# uses img1 and the hourglass model
annotated_img = add_prediction_to_image(img1, bboxes, scores, classes)
plot_images([annotated_img], ['Hourglass 104'],row=1, column=1, fig_size=(10, 10))

"""
Wrap-Up Inference, Annotation, and Plotting
It’s time to put everything together.

The following function accepts an image, the model, and a score threshold. 
The image is forward passed to get all the bounding box predictions. 
These are further filtered by the function filter_detection_on_score and 
annotated by the function add_prediction_to_image. This function returns the image with predictions annotated.
"""
def infer_and_add_prediction_to_image(img, model, score_thresh=0.3):
    prediction = model(np.array([img]))
    bboxes, scores, classes = to_numpy(prediction)
 
    bboxes, scores, classes = filter_detections_on_score(bboxes, scores, classes, 
                                                        score_thresh)
    boxes = normalize_to_pixels_bboxs(bboxes, img)
    img_with_bboxes = add_prediction_to_image(img, boxes, scores, classes)
    return img_with_bboxes

# Uses img1 and the resnet model
annotated_img = infer_and_add_prediction_to_image(img1, resnet)
plot_images([annotated_img], ['ResNet 101'], row=1, column=1, fig_size=(10, 10))

# wrapper function for comparing resnet to hourglass
def show_hourglass_resnet_inference(img, score_thresh=0.3):
    hourglass_infer = infer_and_add_prediction_to_image(img, hourglass)
    resnet_infer = infer_and_add_prediction_to_image(img, resnet)
    image_list = [hourglass_infer, resnet_infer]
    titles = ['Hourglass 104', 'ResNet 101']
    plot_images(image_list, titles, row=1, column=2, fig_size=(20, 10))


show_hourglass_resnet_inference(img1)
show_hourglass_resnet_inference(img2)
show_hourglass_resnet_inference(img3)
show_hourglass_resnet_inference(img4)
