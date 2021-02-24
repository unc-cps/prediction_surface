#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:29:46 2021

@author: ozgur
"""

import matplotlib.pyplot as plt
import numpy as np
import ntpath
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib.style as style
from glob import glob
import pandas as pd
from tqdm import tqdm
from random import shuffle
from scipy import stats
import yolo_util as yu
from keras.models import load_model
import ml_metrics 

def object_unc_def_worst(fname):
    mc_location_file = 'tmp/kitti/' + ntpath.basename(fname).replace('.png','.npz')
    mc_locations = np.load(mc_location_file)['arr_0']
    data = plt.imread(fname)
    fig, ax = plt.subplots(1,2,sharey=True, figsize=(9,5))
    ax[0].imshow(data, alpha=1.0)
    ax[1].imshow(data, alpha=0.5)
    
    for i in range(mc_locations.shape[0]):
        x1, y1, x2, y2 = mc_locations[i,0:4]
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red',lw=0.5)
        ax[1].add_patch(rect)
        ax[1].scatter(x1,y1, marker='x', c='w')
        ax[1].scatter(x1,y2, marker='x', c='yellow')
        ax[1].scatter(x2,y1, marker='x', c='c')
        ax[1].scatter(x2,y2, marker='x', c='r')
        
    points = mc_locations[:,0:2]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(-90,145,'Corner-1: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    points = mc_locations[:,2:4]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(210,420,'Corner-4: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    points = np.c_[mc_locations[:,0],mc_locations[:,3]]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(210,145,'Corner-2: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    points = np.c_[mc_locations[:,2],mc_locations[:,1]]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(-90,420,'Corner-3: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    for i in range(2):
        ax[i].set_xlim([-100,400])
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])
        ax[i].set_ylim([0,430])
    plt.gca().invert_yaxis()
    ax[0].set_title('Car camera view')
    ax[1].set_title('Object detection predictions')
    plt.savefig("../img/uncertain_pred_worst.pdf", bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
    
def object_unc_def_best(fname):
    mc_location_file = 'tmp/kitti/' + ntpath.basename(fname).replace('.png','.npz')
    mc_locations = np.load(mc_location_file)['arr_0']
    data = plt.imread(fname)
    fig, ax = plt.subplots(1,2,sharey=True, figsize=(9,5))
    ax[0].imshow(data, alpha=1.0)
    ax[1].imshow(data, alpha=0.5)
    
    for i in range(mc_locations.shape[0]):
        x1, y1, x2, y2 = mc_locations[i,0:4]
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False, color='red',lw=0.5)
        ax[1].add_patch(rect)
        ax[1].scatter(x1,y1, marker='x', c='w')
        ax[1].scatter(x1,y2, marker='x', c='yellow')
        ax[1].scatter(x2,y1, marker='x', c='c')
        ax[1].scatter(x2,y2, marker='x', c='r')
        
    points = mc_locations[:,0:2]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(660,145,'Corner-1: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    points = mc_locations[:,2:4]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(930,370,'Corner-4: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    points = np.c_[mc_locations[:,0],mc_locations[:,3]]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(930,145,'Corner-2: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    points = np.c_[mc_locations[:,2],mc_locations[:,1]]
    hull1 = ConvexHull(points)
    ax[1].fill(points[hull1.vertices,0], points[hull1.vertices,1], 'g', alpha=0.3)
    for simplex in hull1.simplices:
        ax[1].plot(points[simplex, 0], points[simplex, 1], 'b-', linewidth=1.0)
    ax[1].text(600,370,'Corner-3: \n' + str(np.round(hull1.area,2)), fontsize=12)
        
    for i in range(2):
        ax[i].set_xlim([650,1150])
        ax[i].get_xaxis().set_ticks([])
        ax[i].get_yaxis().set_ticks([])
        ax[i].set_ylim([0,430])
    plt.gca().invert_yaxis()
    ax[0].set_title('Car camera view')
    ax[1].set_title('Object detection predictions')
    plt.savefig("../img/uncertain_pred_best.pdf", bbox_inches = 'tight', pad_inches = 0)
    
    plt.show()
    
def uncertainty_explain(style_name):
    plt.close('all')
    plt.style.use('default') 
    style.use(style_name)
    
    fig, ax = plt.subplots(1,1, figsize=(7,3.0))
    
    sin_scale = 0.3
    np.random.seed(1)
    x = np.arange(0,12,0.1)   # start,stop,step
    y = np.sin(sin_scale*x)
    ax.plot(x,y,'-', lw=2, c='r')
    
    x1 = np.arange(0.5,2.5,0.05)   # start,stop,step
    y1 = np.sin(sin_scale * x1)
    y1_rand = y1 + np.random.normal(0,0.02,y1.shape)
    ax.scatter(x1, y1_rand, marker='o', c='k', s=10)
    
    x1 = np.arange(9.5,11.5,0.05)   # start,stop,step
    y1 = np.sin(sin_scale * x1)
    y1_rand = y1 + np.random.normal(0,0.2,y1.shape)
    ax.scatter(x1, y1_rand, marker='o', c='k', s=10)
    ax.legend(['Ground truth','Training data'], frameon=True, loc='lower center',
              ncol=2,handlelength=1, fontsize=10)
    
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(  AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(  AutoMinorLocator(5))
    
    ax.axvline(x=0, color='k', linestyle='--', lw=1)
    ax.axvline(x=3, color='k', linestyle='--', lw=1)
    ax.axvline(x=9, color='k', linestyle='--', lw=1)
    ax.axvline(x=12, color='k', linestyle='--', lw=1)
    
    ax.axvspan(0, 3, alpha=0.2, color='green')
    ax.axvspan(3, 9, alpha=0.1, color='blue')
    ax.axvspan(9,12, alpha=0.1, color='red')
    
    ax.set_ylabel(r'$\sin(0.3 \times x)$',fontsize=15)
    ax.set_xlabel(r'$x$',fontsize=20)
    
    
    font = {'family': 'sans-serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }
    
    plt.text(1.5, 0.8,'Low Aleatoric\nUncertainty',font, ha='center' )
    plt.text(6.0, 0.3,'High Epistemic\nUncertainty\n(No Training Data)',font,ha='center' )
    plt.text(10.5, 0.8,'High Aleatoric\nUncertainty',font,ha='center' )
    
    
    plt.savefig('../img/regression_uncertainty_regions.pdf', 
                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
def get_iou_old(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    #print(boxA)
    #print(boxB)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def get_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

        
def get_eval_results(model_name,image_path, image_type, plot_gt=False, T=1, p=0.0):
    iou_threshold = 0.5
    total_hull = 0.0
    f_result_path = 'model_pred_res_no_mc.csv'
    f_out = open(f_result_path,'w')
    search_path = image_path +  '*.' + image_type + '_' + model_name + '_' + str(T) + '_' + str(p) + '.npz'
    print('search_path',search_path)
    files = glob(search_path)
    shuffle(files)
    #print(files)
    for i in tqdm(range(len(files))):
    #for i in range(10):
        fname = files[i]
        png_name = ntpath.basename(fname).replace('_' + str(T) + '_' + str(p) + '.npz','')
        png_name = image_path + png_name
        ground_truth_file = png_name.replace('_' + model_name,'') + '.txt'
        ground_truth = pd.read_csv(ground_truth_file,
                                   names=['class','x1','y1','x2','y2'],
                                   sep=' ')
        mc_locations = np.load(fname)['arr_0']
        max_iou_val_list = []
        max_mAP_val_list = []
        avg_hull = 0.0
        if mc_locations.shape[0]:
            num_of_detected = 0.0
            mc_locations_df = pd.DataFrame(data=mc_locations, columns=['x1','y1','x2','y2','center_x','center_y','label'])
            mc_locations_df['label'] = pd.to_numeric(mc_locations_df['label'], downcast='integer')
            cluster_labels = np.unique(mc_locations_df.label.values)
            total_hull = 0.0
            for c_label in cluster_labels:
                if c_label == -1.0:
                    continue
                
                #print('===== c_label',c_label)
                tmp_df = mc_locations_df.query('label == ' + str(c_label))
                tmp_df.drop(['label'], axis=1, inplace=True)
                avg_locations = tmp_df.values.mean(axis=0)
                x1_avg,y1_avg,x2_avg,y2_avg = avg_locations[0:4]
                x1_avg = np.min([avg_locations[0],avg_locations[2]])
                x2_avg = np.max([avg_locations[0],avg_locations[2]])
                y1_avg = np.min([avg_locations[1],avg_locations[3]])
                y2_avg = np.max([avg_locations[1],avg_locations[3]])
                
                max_mAP = 0.0
                max_iou_val = 0.0
                for index, row in ground_truth.iterrows():
                    x1 = np.min([row['x1'],row['x2']])
                    x2 = np.max([row['x1'],row['x2']])
                    y1 = np.min([row['y1'],row['y2']])
                    y2 = np.max([row['y1'],row['y2']])
                    
                    iou_val = get_iou([x1_avg,y1_avg,x2_avg,y2_avg],[x1,y1,x2,y2])
                    
                    map_list_pred = []
                    map_list_gt = []
                    map_list_pred.append([x1_avg,y1_avg,x2_avg,y2_avg])
                    map_list_gt.append([x1,y1,x2,y2])
                    obj_map = ml_metrics.mapk(map_list_gt, map_list_pred)
                    max_mAP = np.max([max_mAP,obj_map])
                    
                    max_iou_val = np.max([max_iou_val, iou_val]) 
                
                
                max_iou_val_list.append(max_iou_val)
                max_mAP_val_list.append(max_mAP)
                
                mc_locations_obj = tmp_df.values
                if mc_locations_obj.shape[0] > 3:
                    tmp_hull_area = 0.0
                    points = mc_locations_obj[:,0:2]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area
                    
                    points = mc_locations_obj[:,2:4]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area
                        
                    points = np.c_[mc_locations_obj[:,0],mc_locations_obj[:,3]]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area
                        
                    points = np.c_[mc_locations_obj[:,2],mc_locations_obj[:,1]]
                    hull1 = ConvexHull(points)
                    tmp_hull_area += hull1.area
                    if p < 0.4:
                        if tmp_hull_area <= 1000: 
                            total_hull += tmp_hull_area
                            num_of_detected += 1.0
                    else:
                        total_hull += tmp_hull_area
                        num_of_detected += 1.0
            avg_hull = total_hull / len(cluster_labels)
            #print(total_hull,num_of_detected)
            avg_hull = total_hull / (num_of_detected + 1e-20)
                    
            if plot_gt:     
                data = plt.imread(png_name.replace('_' + model_name,''))
                fig, ax = plt.subplots(1,1,sharey=True, figsize=(12,5))
                ax.imshow(data)

                for c_label in cluster_labels:
                    tmp_df = mc_locations_df.query('label == ' + str(c_label))
                    tmp_df.drop(['label'], axis=1, inplace=True)
                    avg_locations = tmp_df.values.mean(axis=0)
                    x1, y1, x2, y2 = avg_locations[0:4]
                    center_x, center_y = avg_locations[4:6]
                    #print('center_x',center_x,'center_y',center_y)
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='red',lw=3)
                    ax.add_patch(rect)
                    ax.scatter(center_x, center_y,marker='o',s=50)
                    
                for index, row in ground_truth.iterrows():
                    x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'],row['y2']
                    width, height = x2 - x1, y2 - y1
                    rect = Rectangle((x1, y1), width, height, fill=False, color='white',lw=3)
                    ax.add_patch(rect)
                plt.title(cluster_labels)
                plt.show()
        max_iou_val_list = np.array(max_iou_val_list)
        detected = (max_iou_val_list >= iou_threshold)
        num_of_detected = max_iou_val_list[detected]
        #print('--->',num_of_detected,max_iou_val_list,detected,ground_truth.shape[0])
        #print('+++>',num_of_detected.shape[0])
        #print('===>',max_mAP_val_list)
        f_out.write(model_name + '\t' + png_name + '\t' + str(ground_truth.shape[0]) + '\t' + str(num_of_detected.shape[0]) + '\t' + str(max_iou_val_list.shape[0]) + '\t' + str(max_iou_val_list).replace('\n',' ') + '\t' + str(avg_hull) + '\n')
    f_out.close()
    
    df_result = pd.read_csv(f_result_path,names=['model','png_name','gt_obj','true_detected','detected','iou_preds','uncertainty'], sep='\t' )
    #df_result = df_result.query("true_detected > 0")
    total_objects = np.sum(df_result.gt_obj.values)
    total_true_detected = np.sum(df_result.true_detected.values)
    total_detected = np.sum(df_result.detected.values)  
    #total_detected = np.min([total_objects,total_detected])
    #total_true_detected = np.min([total_objects,total_true_detected])
    print(total_objects,total_detected,total_true_detected)
    uio_vals = [np.fromstring(v.replace('[','').replace(']',''), sep=' ').mean() for v in df_result.iou_preds.values]
    uio_vals = np.array(uio_vals)
    uio_vals[np.isnan(uio_vals)] = 0
    
    avg_iou = uio_vals.mean()
    #print('uio_vals',uio_vals,iou_threshold)
    #print((uio_vals >= iou_threshold))
    avg_iou = uio_vals[(uio_vals >= iou_threshold)].mean()
    TP = total_true_detected
    precision = total_true_detected/total_detected
    recall = total_true_detected/total_objects
    f1 = stats.hmean([precision, recall] )
    uncertainty = np.sum(df_result.uncertainty.values) / total_detected
    print('Avg IoU:',avg_iou)
    print ('TP:',TP)
    print('All Detections:',total_detected)
    print('Precision:',total_true_detected/total_detected)
    print('GT',total_objects)
    print('Recall:',recall)
    print('F_1', f1)
    return avg_iou, TP,precision,recall,f1, uncertainty

def recall_problem():
    # TODO: Set the path to the `.h5` file of the model to be loaded.
    MODEL_PATH = yu.MODEL_PATH + 'yolo-keras.h5'
    
    IMAGE_PATH = '/Volumes/Backup Plus/ds/autonomous_cars/KITTI/paper_format/'
    
    #yu.convert_model_to_keras(output_model=MODEL_PATH,  mc_dropout=False)
    model = load_model(MODEL_PATH)
    
    fname = IMAGE_PATH + '000487.png'
    ground_truth_file = fname + '.txt'
    ground_truth = pd.read_csv(ground_truth_file,
                               names=['class','x1','y1','x2','y2'],
                               sep=' ')
    v_boxes, v_labels, v_scores = yu.get_objects(fname, model)
    print(v_boxes)
    mc_locations = []
    
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        width, height = x2 - x1, y2 - y1
        mc_locations.append(np.array([x1,y1,x2,y2,x1+width/2,y1+height/2]))
        
    mc_locations = np.array(mc_locations)
    data = plt.imread(fname)
    fig, ax = plt.subplots(1,1,sharey=True, figsize=(12,5))
    ax.imshow(data)
    for i in range(mc_locations.shape[0]):
        x1, y1, x2, y2 = mc_locations[i,0:4]
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False,lw=3,ls='--', color='red')
        ax.add_patch(rect)
        ax.scatter(mc_locations[i,4],mc_locations[i,5], marker='x',
                   c='g',s=150)
        
    for index, row in ground_truth.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'],row['y2']
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, fill=False,lw=3,ls='--', color='white',zorder=10)
        ax.add_patch(rect)
    plt.savefig("../img/yolo_recall_problem.png", bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
if __name__ == "__main__":
    #recall_problem()

    IMAGE_PATH = '/Volumes/Backup Plus/ds/autonomous_cars/stanford_cars_dataset/paper_format/'
    IMAGE_TYPE = 'jpg'
    IMAGE_NAME = 'stanford_cars'
    model_name = 'yolo'
    #object_unc_def_worst(KITTI_IMAGE_PATH + 'data_object_image_2/training/image_2/006184.png')
    #object_unc_def_best(KITTI_IMAGE_PATH + 'data_object_image_2/training/image_2/001018.png')
    #uncertainty_explain('bmh')
    p_size_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.65]
    p_size_list = np.linspace(0.1,0.5,9).round(2)
    p_size_list = np.linspace(0.1,0.5,17).round(2)
    
    #p_size_list = [0.1]
    #p_size_list = p_size_list[[1,3,5,7,9,11,13,15]]
    avg_iou_list = []
    TP_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    unc_list = []
    for p_size in p_size_list:
        avg_iou, TP,precision,recall,f1,unc = get_eval_results(model_name,IMAGE_PATH,IMAGE_TYPE, 
                                                           plot_gt=False, T=20, p=p_size)
        avg_iou_list.append(avg_iou)
        TP_list.append(TP)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        unc_list.append(unc)
    plt.close('all')
    plt.style.use('default') 
    plt.style.use('fivethirtyeight') 
    plt.plot(p_size_list,unc_list)
    plt.ylabel('Prediction Surface')
    plt.xlabel(r'Dropout ($p$)')
    plt.savefig('../img/' + model_name + '-' + IMAGE_NAME + '-unc.pdf', format='pdf',
                bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    plt.scatter(unc_list,precision_list,c='r',marker='+')
    #plt.scatter(unc_list,recall_list,c='b',marker='+')
    #plt.scatter(unc_list,f1_list,c='k',marker='x')
    plt.scatter(unc_list,avg_iou_list,c='c',marker='x')
    
    z = np.polyfit(unc_list, precision_list, 1)
    p = np.poly1d(z)
    plt.plot(unc_list,p(unc_list),"r--",lw=1)

    z = np.polyfit(unc_list, avg_iou_list, 1)
    p = np.poly1d(z)
    plt.plot(unc_list,p(unc_list),"c--",lw=1)
    
    plt.legend(['Precision','IOU'],ncol=4, loc='upper center',
               bbox_to_anchor=(0.50, 1.15),shadow=True,frameon=True)
    plt.xlabel('Prediction Surface')
    plt.ylabel('Values')
    
    plt.savefig('../img/' + model_name + '-' + IMAGE_NAME + '-perf.pdf', format='pdf',
                bbox_inches = 'tight', pad_inches = 0)
    #if IMAGE_NAME == 'kitti' and model_name =='yolo':
    #    plt.xlim(0,500)
    plt.show()
    
    
    
    