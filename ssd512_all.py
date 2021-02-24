#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:44:14 2021

@author: ozgur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 16:17:58 2021

@author: ozgur
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 15:11:03 2021

@author: ozgur
"""
import os,sys,glob,ntpath
from keras import backend as K
import pandas as pd
from random import shuffle

ROOT_DIR = os.path.abspath("ssd_keras")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Set the image size.
img_height = 512
img_width = 512

# TODO: Set the path to the `.h5` file of the model to be loaded.
weights_path = '/Volumes/Backup Plus/ds/models/VGG_VOC0712_SSD_512x512_iter_120000.h5'

IMAGE_PATH = '/Volumes/Backup Plus/ds/autonomous_cars/KITTI/paper_format/'
IMAGE_TYPE = 'png'
IMAGE_NAME = 'kitti'

classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm
from matplotlib.patches import Rectangle
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

from models.keras_ssd512 import ssd_512
from keras_loss_function.keras_ssd_loss import SSDLoss

def get_model(p_size,mc_dropout=True):
    K.clear_session() # Clear previous models from memory.

    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05], # The scales for MS COCO are [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                   two_boxes_for_ar1=True,
                   steps=[8, 16, 32, 64, 128, 256, 512],
                   offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                   clip_boxes=False,
                   variances=[0.1, 0.1, 0.2, 0.2],
                   normalize_coords=True,
                   subtract_mean=[123, 117, 104],
                   swap_channels=[2, 1, 0],
                   confidence_thresh=0.5,
                   iou_threshold=0.45,
                   top_k=200,
                   nms_max_output_size=400,
                   mc_dropout=mc_dropout,
                   dropout_size=p_size)
    
    # 2: Load the trained weights into the model.
    model.load_weights(weights_path, by_name=True)
    # 3: Compile the model so that Keras won't complain the next time you load it.
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    return model

def get_pred_uncertainty(fname, model, mc_dropout_size=20, 
                         plot_ground_truth=False, mc_dropout=True):
    ground_truth_file = fname + '.txt'
    ground_truth = pd.read_csv(ground_truth_file,
                               names=['class','x1','y1','x2','y2'],
                               sep=' ')
    
    orig_images = [] # Store the images here.
    input_images = [] # Store resized versions of the images here.

    orig_images.append(imread(fname))
    img = image.load_img(fname, target_size=(img_height, img_width))
    img = image.img_to_array(img) 
    input_images.append(img)
    input_images = np.array(input_images)
    
    mc_locations = []
    # for _ in tqdm(range(mc_dropout_size)):
    for _ in range(mc_dropout_size):
        y_pred = model.predict(input_images)
        confidence_threshold = 0.5
        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
        
        for box in y_pred_thresh[0]:
            x1 = box[2] * orig_images[0].shape[1] / img_width
            y1 = box[3] * orig_images[0].shape[0] / img_height
            x2 = box[4] * orig_images[0].shape[1] / img_width
            y2 = box[5] * orig_images[0].shape[0] / img_height
            width, height = x2 - x1, y2 - y1
            mc_locations.append(np.array([x1,y1,x2,y2,x1+width/2,y1+height/2]))
            
    mc_locations = np.array(mc_locations)
    avg_surface = -1.0
    if mc_locations.shape[0]:
        clustering = DBSCAN(eps=100, min_samples=2).fit(mc_locations)
        mc_locations = np.c_[mc_locations,clustering.labels_.ravel()]
        
        mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1','y1','x2','y2','center_x','center_y','label'])
        cluster_labels = np.unique(mc_locations[:,6])
        total_cluster_surface = 0.0
        if mc_dropout == True:
            for cluster_label in cluster_labels:
                cluster_df = mc_locations_df.query('label == ' + str(cluster_label))
                if cluster_df.shape[0] > 2:
                    center_data = cluster_df[['x1','y1']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x2','y1']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x1','y2']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x2','y2']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                avg_surface = total_cluster_surface/mc_locations.shape[0]
    
    if plot_ground_truth:
        data = plt.imread(fname)
        fig, ax = plt.subplots(1,1,sharey=True, figsize=(12,5))
        ax.imshow(data)
        for i in range(mc_locations.shape[0]):
            x1, y1, x2, y2 = mc_locations[i,0:4]
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
            ax.scatter(mc_locations[i,4],mc_locations[i,5], marker='x',
                       c='g',s=150)
            
        for index, row in ground_truth.iterrows():
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'],row['y2']
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='white')
            ax.add_patch(rect)
        plt.show()
    
    return mc_locations, avg_surface

def find_uncertainties_for_all(files, mc_dropout=True):
    p_size_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
    p_size_list = np.linspace(0.1,0.7,13).round(2)
    p_size_list = np.linspace(0.1,0.5,17).round(2)
    #p_size_list = p_size_list[[1,3,5,7,9,11,13,15]]
    #p_size_list = [0.2]
    shuffle(p_size_list)
    #p_size_list = [0.0]
    if mc_dropout == False:
        mc_dropout_size = 1
    else:
        mc_dropout_size = 20
    for p_size in p_size_list:
        model = get_model(p_size, mc_dropout=mc_dropout)
        print('Start for p:', p_size)
        rand_idx = np.random.randint(0,len(files),size=5)
        avg_surface_list = []
        image_name_list = []
        for i in tqdm(range(len(rand_idx))):
            idx = rand_idx[i]
    
            fname = files[idx]
            mc_locations, avg_surface = get_pred_uncertainty(fname, model, 
                                                             mc_dropout_size=mc_dropout_size,
                                                             plot_ground_truth=False,
                                                             mc_dropout=mc_dropout)
            
            image_name_list.append(fname)
            avg_surface_list.append(avg_surface)
            mc_location_file = fname + '_ssd512_' + str(mc_dropout_size) + '_' + str(p_size) +  '.npz'
            np.savez(mc_location_file,mc_locations)
        
        unc_df = pd.DataFrame({'image_name':image_name_list,'avg_surface':avg_surface_list})
        unc_df.to_csv('ssd512_' + IMAGE_NAME + '.csv', index=False)
    
def create_df_from_folder():
    npz_list = glob.glob('tmp/ssd512/' + IMAGE_NAME + '/*.npz')
    avg_uncs_list = []
    png_file_list = []
    for npz_path in npz_list:
        mc_locations = np.load(npz_path)['arr_0']
        avg_surface = -1.0
        if mc_locations.shape[0]:
            clustering = DBSCAN(eps=200, min_samples=2).fit(mc_locations)
            mc_locations = np.c_[mc_locations,clustering.labels_.ravel()]
            mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1','y1','x2','y2','center_x','center_y','label','label2'])
            cluster_labels = np.unique(mc_locations[:,6])
            total_cluster_surface = 0.0
            for cluster_label in cluster_labels:
                cluster_df = mc_locations_df.query('label == ' + str(cluster_label))
                if cluster_df.shape[0] > 2:
                    center_data = cluster_df[['x1','y1']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x2','y1']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x1','y2']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                    
                    center_data = cluster_df[['x2','y2']].values
                    hull = ConvexHull(center_data)
                    total_cluster_surface += hull.area
                avg_surface = total_cluster_surface/mc_locations.shape[0]
        avg_uncs_list.append(avg_surface)
        png_file = IMAGE_PATH + ntpath.basename(npz_path).replace('.npz','.' + IMAGE_TYPE)
        png_file_list.append(png_file)
    df = pd.DataFrame({'image_name':png_file_list,'avg_surface':avg_uncs_list})
    df.to_csv('ssd512_' + IMAGE_NAME + '.csv', index=False)
    return df
    
def plot_highly_uncertain_imgs(n=1, from_folder=False):
    if from_folder:
        unc_df = create_df_from_folder()
    else:
        unc_df = pd.read_csv('ssd512' + IMAGE_NAME + '.csv')
    unc_df = unc_df.query('avg_surface > 0.0')
    unc_df.sort_values(['avg_surface'],ascending=False,inplace=True)
    print(unc_df)
    for index, row in unc_df.iterrows():
        npz_file = ntpath.basename(row['image_name']).replace('.png','.npz')
        mc_locations = np.load('tmp/ssd512/' + IMAGE_NAME + '/' + npz_file)['arr_0']
        data = plt.imread(row['image_name'])
        fig, ax = plt.subplots(1,1,sharey=True, figsize=(12,5))
        ax.imshow(data)
        for i in range(mc_locations.shape[0]):
            x1, y1, x2, y2 = mc_locations[i,0:4]
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
        plt.title(row['image_name'])
        plt.show()
        
        

if __name__ == "__main__":
    files = glob.glob(IMAGE_PATH + '**/*.' + IMAGE_TYPE, recursive=True)
    for _ in range(10000):
        find_uncertainties_for_all(files, mc_dropout=True)
    #plot_highly_uncertain_imgs(from_folder=True)
    
    
    