import os
import cv2
import numpy as np
import math
import time
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization
from keras.layers.merge import concatenate
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from models import *
import argparse

def str_add(str_list, add_str):
    new_str_list = []
    for i in range(len(str_list)):
        bn, ext = os.path.splitext(str_list[i])
        new_str_list.append(os.path.basename(bn) + add_str + ext)
    return new_str_list

def write_dsc(filename, img_filename, label_imgs, compare_imgs):
    if len(label_imgs) != len(compare_imgs):
        return 0.0
    
    file = open(filename, 'w')
    file.write('File name,Precision,Recall,IoU,Dice\n')
    dices = []
    ious = []
    precisions = []
    recalls = []
    count = 0
    for i in range(0, len(label_imgs)):
        # foreground, background
        norm_label_img = np.asarray((label_imgs[i] / 255), dtype='int64')
        if np.sum(norm_label_img) == 0:
            continue

        norm_compare_img = np.asarray((compare_imgs[i] / 255), dtype='int64')
        sum_pixel_tgt = np.sum(norm_compare_img)

        img_size = (norm_label_img.shape[0]*norm_label_img.shape[1])
        FPFN = abs(abs(norm_label_img-norm_compare_img))
        TPTN = abs(FPFN-1)
        TP = np.asarray(np.logical_and(np.asarray(norm_label_img, dtype='bool'), np.asarray(norm_compare_img, dtype='bool')),dtype='int64')
        TN = abs(TPTN-TP)
        FP = abs(norm_label_img-abs(TN-1))
        FN = abs(FPFN-FP)
        # print(np.sum(TP),np.sum(TN),np.sum(FP),np.sum(FN))

        precision = 0
        if np.sum(TP) == 0 and np.sum(FP) == 0:
            precision = 0
        else:
            precision = np.sum(TP)/(np.sum(TP)+np.sum(FP))
        precisions.append(precision)
        
        recall = 0
        if np.sum(TP) == 0 and np.sum(FN) == 0:
            recall = 0
        else:
            recall = np.sum(TP)/(np.sum(TP)+np.sum(FN))
        recalls.append(recall)

        iou = 0
        if (np.sum(norm_label_img)+np.sum(norm_compare_img)-np.sum(TP)) == 0:
            iou = 0
        else:
            iou = np.sum(TP)/ (np.sum(norm_label_img)+np.sum(norm_compare_img)-np.sum(TP))
        ious.append(iou)

        if precision == 0 and recall == 0:
            dice = 0
        else:
            dice = recall*precision*2 / (recall+precision)
        dices.append(dice)
        
        file.write(img_filename[i] + ',' + str(precision) + ',' + str(recall) + ',' + str(iou) + ',' + str(dice) + '\n')
        count+=1
    
    ave_dice = np.sum(dices) / count
    ave_precision = np.sum(precisions) / count
    ave_iou = np.sum(ious) / count
    ave_recall = np.sum(recalls) / count
    
    file.write('\naverage,' + str(ave_precision) + ',' + str(ave_recall) + ',' + str(ave_iou) + ',' + str(ave_dice))
    file.close()

def load_label(inputpath):
    imglist = []

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue

            filename = os.path.join(root, fn)
            testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            imglist.append(testimage)

    imgsdata = np.asarray(imglist, dtype=np.float32)

    return imgsdata, sorted(files)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Instruments Segmentation')
    parser.add_argument('--data_type', '-d', type=str, default='rgb')
    parser.add_argument('--case', '-c', type=str, default='1')
    parser.add_argument('--depth_type', '-depth', type=str, default='estimated_depth')
    parser.add_argument('--gpus', '-g', type=int, default=1)
    args = parser.parse_args()

    csv_filepath = './data/results/'+args.data_type+'/'+args.case+'/pred.csv'
    dir_results = './data/results/'+args.data_type+'/'+args.case+'/'

    if args.data_type == 'rgb':
        weight_path = './weights/'+args.data_type+'_'+args.case+'_bestmodel.hdf5'
        net = Network()
        model = net.make_ternausNet16_paper(label_num=2, pre_train_model=weight_path, is_trainable=True, gpus=args.gpus)

    elif args.data_type=='rgbd':
        weight_path = './weights/'+args.data_type+'_'+args.depth_type+'_'+args.case+'_bestmodel.hdf5'
        net = Network()
        model = net.make_ternausNet16_rgbd(label_num=2, pre_train_model=weight_path, is_trainable=True, gpus=args.gpus)
        dir_test_depth = './data/'+args.case+'/test/' +args.depth_type+ '/'
        depth_test, depth_test_filename = load_images(dir_test_depth, DEPTH_SIZE, 'Gray')
        depth_test = np.asarray(depth_test, dtype=np.float32)
        depth_ave_file = './data/'+args.case+"/depth_avg.txt"
        with open(depth_ave_file) as f:
            depth_ave = f.readline()
            print(depth_ave)
        depth_test -= float(depth_ave)# depth average each scene
        csv_filepath = './data/results/'+args.depth_type+'/'+args.case+'/pred.csv'
        dir_results = './data/results/'+args.depth_type+'/'+args.case+'/'

    dir_test_rgb = './data/'+args.case+'/test/rgb/'
    dir_test_label = './data/'+args.case+'/test/label/'
    image_test, image_test_filenames = load_images(dir_test_rgb, IMAGE_SIZE, 'Color', preprocessing='Imagenet')
    label_test, label_filenames = load_label(dir_test_label)
    label_test = (label_test > 10) * 255
    print(len(image_test))

    results = []
    
    t1 = time.time()
    for i in range(len(image_test)):
        if args.data_type=="rgb":
            temp = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            temp[0] = image_test[i]
            result = model.predict([temp], verbose=0)
            instrument = result[0, :, :, 0]
            background = result[0, :, :, 1]

            instrument = np.asarray([instrument], dtype=np.float64)
            instrument = np.asarray(instrument, dtype=np.float64).reshape((1, IMAGE_SIZE, IMAGE_SIZE))
            instrument = instrument.transpose(1, 2, 0)
            
        elif args.data_type=="rgbd":
            temp = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3))
            temp[0] = image_test[i]
            temp_depth = np.zeros((1, DEPTH_SIZE, DEPTH_SIZE, 1))
            temp_depth[0] = depth_test[i]
            temp_depth[0] -= float(depth_ave)
            temp_depth[0] -= float(depth_ave)
            temp_depth[0] -= float(depth_ave)
            temp_depth[0] -= float(depth_ave)
            result = model.predict([temp, temp_depth], verbose=0)
            instrument = result[0, :, :, 0]
            background = result[0, :, :, 1]

            instrument = np.asarray([instrument], dtype=np.float64)
            instrument = np.asarray(instrument, dtype=np.float64).reshape((1, IMAGE_SIZE, IMAGE_SIZE))
            instrument = instrument.transpose(1, 2, 0)

        results.append(instrument)
        print('\r{0:d} / {1:d}'.format(i+1, len(image_test)), end='')
    print('estimation time per frame: ', (time.time()-t1)/len(results))
    results = np.asarray(results, dtype = 'float64')

    height = label_test[0].shape[0]
    width = label_test[0].shape[1]
    results *= 255.0
    results_th = []
    for i in range(len(results)):
        ret, th = cv2.threshold(results[i], 127, 255, cv2.THRESH_BINARY)
        th = cv2.resize(th, (width, height))
        results_th.append(th)

    if(not os.path.isdir(dir_results)):
        os.makedirs(dir_results)

    output_name_list = str_add(image_test_filenames, "output")
    write_dsc(csv_filepath, output_name_list ,label_test, results_th)
    save_images(dir_results, output_name_list, results_th)