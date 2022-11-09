# import packages
import os
import gc
from secrets import choice
import sys

import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import skimage
from skimage.feature import hog, canny
from skimage.filters import sobel
from skimage import color

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# from keras import layers
# import keras.backend as K
# from keras.models import Sequential, Model
# from keras.preprocessing import image
# from keras.layers import Input, Dense, Activation, Dropout
# from keras.layers import Flatten, BatchNormalization
# from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D 
# from keras.applications.imagenet_utils import preprocess_input
# from xgboost import train
# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications import ResNet50
# from tf_explain.core.activations import ExtractActivations
# from tf_explain.core.grad_cam import GradCAM
# from sklearn.model_selection import train_test_split
# from keras.utils.data_utils import get_file

from PIL import Image
from tqdm import tqdm #shows a progress meter on looping structures
import random as rnd
import cv2
# from keras.preprocessing.image import ImageDataGenerator
# from numpy import expand_dims
# from livelossplot import PlotLossesKeras

# load data
path = '/Users/tawate/Library/CloudStorage/OneDrive-SAS/08_CDT_DataScience/state-farm-distracted-driver-detection'
train_df = pd.read_csv(path + '/driver_imgs_list.csv')
train_df['path'] = path + '/imgs/train/' + train_df['classname']+ '/' +train_df['img']
pred_df = pd.read_csv(path + '/sample_submission.csv')

# define class names
classes = { 'c0' : 'normal driving',
            'c1' : 'texting - right',
            'c2' : 'talking on the phone - right',
            'c3' : 'texting - left',
            'c4' : 'talking on the phone - left',
            'c5' : 'operating the radio',
            'c6' : 'drinking',
            'c7' : 'reaching behind',
            'c8' : 'hair and makeup',
            'c9' : 'talking to passenger'}

train_df.head(10)
pred_df.head(10)

# count by classes
print('Class Count: ', len(train_df['classname'].value_counts()))
train_df['classname'].value_counts()

# check for missing data
train_df.isna().sum()

""" Visualization (EDA) """
# plot one random image from each class
plt.figure(figsize=(15,12))
for idx, i in enumerate(train_df.classname.unique()):
    plt.subplot(4,7,idx+1)
    df = train_df[train_df['classname'] == i].reset_index(drop=True)
    image_path = df.loc[rnd.randint(0,len(df))-1,'path']
    img = Image.open(image_path)
    img = img.resize((224,224))
    plt.imshow(img)
    plt.axis('off')
    plt.title(classes[i])
plt.tight_layout()
plt.show()

# plot of images for each class
def plot_species(df, class_name):
    plt.figure(figsize = (12,12))
    classes_df = df[df['classname'] == class_name].reset_index(drop = True)
    plt.suptitle(classes[class_name])
    for idx, i in enumerate(np.random.choice(classes_df['path'],32)):
        plt.subplot(8,8,idx+1)
        image_path = i
        img = Image.open(image_path)
        img = img.resize((224,224))
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout
    plt.show()

for class_ in train_df['classname'].unique():
    plot_species(train_df,class_)

""" Class distribution analysis """
plot = sns.countplot(x = train_df['classname'],color = '#2596be')
sns.set(rc={'figure.figsize':(30,25)})
sns.despine()

# percent breakdown
plt.figure(figsize=(5,5))
class_cnt = train_df.groupby(['classname']).size().reset_index(name = 'counts')
colors = sns.color_palette('Paired')[0:9]
plt.pie(class_cnt['counts'], labels=class_cnt['classname'], colors=colors, autopct='%1.1f%%')
plt.legend(loc='upper right')
plt.show()


""" Image resolutions """
widths, heights = [], []
for path in tqdm(train_df["path"]):
    width, height = Image.open(path).size
    widths.append(width)
    heights.append(height)

train_df["width"] = widths
train_df["height"] = heights
train_df["dimension"] = train_df["width"] * train_df["height"]
train_df.sort_values('width').head(100)

""" Color Analysis """
# gives us an idea of the augmentation technique to use
def is_grey_scale(givenImage):
    w,h = givenImage.size
    for i in range(w):
        for j in range(h):
            r,g,b = givenImage.getpixel((i,j))
            if r != g != b: return False
    return True

# check color scheme of train images
sampleFrac = .5
isGreyList = []
for imageName in train_df['path'].sample(frac=sampleFrac):
    val = Image.open(imageName).convert('RGB')
    isGreyList.append(is_grey_scale(val))
print(np.sum(isGreyList) / len(isGreyList))

# get mean intensity for each channel (RGB)
def get_rgb_mean(row):
    img = cv2.imread(row['path'])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.sum(img[:,:,0]), np.sum(img[:,:,1]), np.sum(img[:,:,2])

tqdm.pandas()
train_df['R'], train_df['G'], train_df['B'] = zip(*train_df.progress_apply(lambda row: get_rgb_mean(row), axis=1))

# build color distributions
def show_color_dist(df, count):
    fig, axr = plt.subplots(count, 2, figsize=(15, 15))
    if df.empty:
        print("Image intensity of selected color is weak")
        return
    for idx, i in enumerate(np.random.choice(df['path'], count)):
        img = cv2.imread(i)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axr[idx, 0].imshow(img)
        axr[idx, 0].axis('off')
        axr[idx, 1].set_title('R={:.0f}, G={:.0f}, B={:.0f} '.format(np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])))
        x, y = np.histogram(img[:,:,0], bins = 255)
        axr[idx,1].bar(y[:-1], x, label='R', alpha=0.8, color='red')
        x, y = np.histogram(img[:,:,1], bins=255)
        axr[idx,1].bar(y[:-1], x, label='G', alpha=0.8, color='green')
        x, y = np.histogram(img[:,:,2], bins=255)
        axr[idx,1].bar(y[:-1], x, label='B', alpha=0.8, color='blue')
        axr[idx,1].legend()
        axr[idx,1].axis('off')

# red images and their color distribution
df = train_df[((train_df['B']) < train_df['R']) & ((train_df['G']) < train_df['R'])]
if df.size != 0:
    show_color_dist(df, 8)

# green imaes and their color distribution
df = train_df[(train_df['G'] > train_df['R']) & (train_df['G'] > train_df['B'])]
if df.size != 0:
    show_color_dist(df, 8)

# blue images and their color distribution
df = train_df[(train_df['B'] > train_df['R']) & (train_df['B'] > train_df['G'])]
if df.size != 0:
    show_color_dist(df, 8)

""" Analyzing Edes 
sobel filter: 
    1. is a way of getting basic edge magnitude/gradient image
    2. gradient method: looks for the max and min of the first derivative of the image pixels. 
    (large rate of change in intensity from pixel to pixel)
    3. calculates gradient at each pixel. Finds the direction of the largest gradient
    4. Uses 2 3x3 kernels (matrics convolved with an image A (matrix of pixels)). 1 kernel for horizontal 
    and 1 for vertical
    5. its larger convolution kernel smooths the input image to a greater extent and so makes the operator 
    less sensitive to noise. The operator also enerally produces considerably higher output values for 
    similar edges, compared with the Roberts Cross.
    6. https://www.cs.auckland.ac.nz/compsci373s1c/PatricesLectures/Edge%20detection-Sobel_2up.pdf
"""

def edge_images_gray(class_name):
    classes_df = train_df[train_df['classname'] == class_name].reset_index(drop=True)
    for idx, i in enumerate(np.random.choice(classes_df['path'],2)):
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = sobel(image)
        gray_edges = sobel(gray)
        dimension = edges.shape
        fig = plt.figure(figsize=(8,8))
        plt.suptitle(classes[class_name])
        plt.subplot(2,2,1)
        plt.imshow(gray_edges)
        plt.subplot(2,2,2)
        plt.imshow(edges[:dimension[0],:dimension[1],0], cmap="gray")
        plt.subplot(2,2,3)
        plt.imshow(edges[:dimension[0],:dimension[1],1], cmap='gray')
        plt.subplot(2,2,4)
        plt.imshow(edges[:dimension[0],:dimension[1],2], cmap='gray')
        plt.show()

for class_name in train_df['classname'].unique():
    edge_images_gray(class_name)


""" HSV Tranformation
    Since this is about time series ordering, transfering to HSV space will be useful for identfying
    shadows and illumination. 
    HSV = Hue Saturation Value
        1. Best for segmenting based on color differences. 
        2. Hue [0-179]: represents color
        3. Saturation [0-255]: represents amount to which the color mixes with white (shade)
        4. Value [0-255]: represents amount to which the color mixes with black (brighness)
"""
def hsv_images(class_name):
    class_df = train_df[train_df['classname'] == class_name].reset_index(drop=True)
    for idx, i in enumerate(np.random.choice(train_df['path'], 2)):
        image = cv2.imread(i)
        hsv = color.rgb2hsv(image)
        dimension = hsv.shape
        print(dimension)
        fig = plt.figure(figsize=(8, 8))
        plt.suptitle(classes[class_name])
        plt.subplot(2,2,1)
        plt.imshow(image)
        plt.subplot(2,2,2)
        plt.imshow(hsv[:dimension[0],:dimension[1],0], cmap="PuBuGn")
        plt.subplot(2,2,3)
        plt.imshow(hsv[:dimension[0],:dimension[1],1], cmap='PuBuGn')
        plt.subplot(2,2,4)
        plt.imshow(hsv[:dimension[0],:dimension[1],2], cmap='PuBuGn')
        plt.show()
    
for class_name in train_df['classname'].unique():
    hsv_images(class_name)

""" Corner detection
    - really good example of corner detection broken down https://sbme-tutorials.github.io/2018/cv/notes/6_week6.html#:~:text=Corner%20Detector%20using%20eigen%20values,-Getting%20the%20two&text=If%20both%20values%20are%20small,so%20we%20have%20a%20corner.
    - goodFeaturesToTrack: https://www.geeksforgeeks.org/python-corner-detection-with-shi-tomasi-corner-detection-method-using-opencv/
        - looks for significant change (max gradient) in all directions
        - edges appear where one eigen value is much greater than the other
        - corners appear where both eigen values are large and relatively similar to eachother
        - must be a gray scale image
        - steps:
            - finds the corner quality measure at every pixel using cornerMinEigenVal()
            - performs non-maximum suppression (see below)
            - corners with min eigenvalue < qualityLevel  * max_qualityMeasureMap(x,y) are rejected
            - remaining corners are sorted by quality measure in descending order
            - remove each corner for which there is a stronger corner at a distance > minDistance (euclidean distance)
        - cornerMinEigenVal()
            - same as corner_EigenValsAndVecs, but only stores the minimal eigenvalue. min(eigen_1, eigen_2)
            - since algorithm is scanning entire image window for largest pixel intensity gradients, this can be a computationally heavy process
            - above function uses taylor expansion to simplify scoring function R = min(eigen_val_1, eigen_val_2)
        - cornerEigenValsAndVecs()
            - every pixel consider a block size x block size  neighborhood (S_p)
            - calculate covariation matrix of derivatives of pixel block itensity 
                - M = [ sum((DI/Dx))^2 , sum(DI/Dx*DI/Dy)  
                        sum(DI/Dx*DI/Dy), sum((DI/Dy)^2)]
                - output is stored as (eigen_1, eigen_2, x1, y1, x2, y2)
                    - eigen_1 and eigen_2 = eigenvalues of M
                    - (x1, y1) = eigenvectors of eigen_1
                    - (x2, y2) = eigenvectors of eigen_2
        - need to review Eigenvalue, eigenvectors, sobel operators, and covariance matricies
"""
def corner_images_gray(class_name):
    classes_df = train_df[train_df['classname'] == class_name].reset_index(drop = True)
    for idx, i in enumerate(np.random.choice(classes_df['path'], 4)):
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners_gray = cv2.goodFeaturesToTrack(gray, maxCorners=50, qualityLevel=0.02, minDistance=20)
        corners_gray = np.float32(corners_gray)
        for item in corners_gray:
            x,y = item[0]
            cv2.circle(image, (int(x), int(y)), 6, (0,255,0), -1)
        fig = plt.figure(figsize=(16,16))
        plt.suptitle(classes[class_name])
        plt.subplot(2,2,1)
        plt.imshow(image, cmap="BuGn")
        plt.show()

for class_name in train_df['classname'].unique():
    corner_images_gray(class_name)

""" Sift Features
    Look into the functionality of SIFT features
"""
def sift_images_gray(class_name):
    classes_df = train_df[train_df['classname'] == class_name].reset_index(drop = True)
    for idx, i in enumerate(np.random.choice(classes_df['path'], 4)):
        image = cv2.imread(i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        kp_img = cv2.drawKeypoints(image, kp, None, color = (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        fig = plt.figure(figsize=(16, 16))
        plt.suptitle(classes[class_name])
        plt.subplot(2,2,1)
        plt.imshow(kp_img, cmap="viridis")
        plt.show()

for class_name in train_df['classname'].unique():
    sift_images_gray(class_name)
