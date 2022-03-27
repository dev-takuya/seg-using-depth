import os
import cv2
import numpy as np
from keras import *
from keras.layers import *
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.applications import vgg16, vgg19
from keras.utils import multi_gpu_model

IMAGE_SIZE = 1024
DEPTH_SIZE = 128
CHANNELS = 3
SEGMENTATION_TARGET_LABEL = 255
BATCH_SIZE = 2
EPOCHS = 30

# scene_index='1'
# dir_test_rgb = './data/' + scene_index + '/test/estimated_depth_fcrn/'
# dir_test_rgb = './data/' + scene_index + '/test/rgb/'
# dir_test_label = './data/' + scene_index + '/test/label/'

class Dataset():
    def __init__(self, preprocessing='None', scene_index='1', data_type='rgb', depth_type='estimated_depth'):
        print('make dataset')
        print('validation type: '+scene_index)

        if data_type=='rgbd':
            dir_train_depth = './data/' + scene_index + '/train/'+depth_type+'/'
            dir_val_depth = './data/' + scene_index + '/val/'+depth_type+'/'
            self.train_depth, self.train_depth_filename = load_images(dir_train_depth, DEPTH_SIZE, 'Gray')
            self.val_depth, self.val_depth_filename = load_images(dir_val_depth, DEPTH_SIZE, 'Gray')
            depth_ave = np.average(self.train_depth)
            print('depth average: ' + str(depth_ave))
            self.train_depth -= depth_ave
            self.val_depth -= depth_ave
            with open('./data/'+scene_index+'/depth_avg.txt', mode='w') as f:
                s = f.write(str(depth_ave))
            print('train depth: ', self.train_depth.shape)
            print('val depth: ', self.val_depth.shape)
        
        dir_train_rgb = './data/' + scene_index + '/train/rgb/'
        dir_train_label = './data/' + scene_index + '/train/label/'
        self.train_image, self.train_image_filename = load_images(dir_train_rgb, IMAGE_SIZE, 'Color', preprocessing)
        self.train_label, self.train_label_filename = load_images(dir_train_label, IMAGE_SIZE, 'Gray')

        dir_val_rgb = './data/' + scene_index + '/val/rgb/'
        dir_val_label = './data/' + scene_index + '/val/label/'
        self.val_image, self.val_image_filename = load_images(dir_val_rgb, IMAGE_SIZE, 'Color', preprocessing)
        self.val_label, self.val_label_filename = load_images(dir_val_label, IMAGE_SIZE, 'Gray')

        self.train_label = (self.train_label > 10) * 255.0
        self.val_label = (self.val_label > 10) * 255.0
        
        self.train_label /= np.max(self.train_label)
        self.val_label /= np.max(self.val_label)

        self.train_label = make_binary_label(self.train_label)
        self.val_label = make_binary_label(self.val_label)
        
        print('train img: ', self.train_image.shape)
        print('train label: ', self.train_label.shape)
        print('val img: ', self.val_image.shape)
        print('val label: ', self.val_label.shape)

        print('done')

    def convlabelnum2desire(self, label, orgnum, outnum, w, h):
        for case in range(0, len(label)):
            for i in range(0, h):
                for j in range(0, w):
                    if(label[case][i][j][0] != orgnum):
                        label[case][i][j][0] = 0.0
                    else:
                        label[case][i][j][0] = outnum

class Network():
    def __init__(self, is_train=True):
        print('create model')

    def make_ternausNet16_paper(self, label_num, is_trainable=True, pre_train_model=None, gpus=1):
        print('make model: ternausNet16')
        with tf.device("/cpu:0"):
            input_tensor = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            model_vgg = vgg16.VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

            if is_trainable == False:
                for layer in model_vgg.layers[:15]:
                    layer.trainable = False

            layer_vgg = dict([(layer.name, layer.output) for layer in model_vgg.layers])

            encode1 = layer_vgg['block1_conv2']
            encode2 = layer_vgg['block2_conv2']
            encode3 = layer_vgg['block3_conv3']
            encode4 = layer_vgg['block4_conv3']
            encode5 = layer_vgg['block5_conv3']
            pool5 = layer_vgg['block5_pool']

            up5 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(pool5)
            up5 = BatchNormalization()(up5)
            up5 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(up5)
            merge5 = concatenate([up5, encode5], axis=-1)
            
            up4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(merge5)
            up4 = BatchNormalization()(up4)
            up4 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(up4)
            merge4 = concatenate([up4, encode4], axis=-1)
            
            up3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(merge4)
            up3 = BatchNormalization()(up3)
            up3 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(up3)
            merge3 = concatenate([up3, encode3], axis=-1)
            
            up2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(merge3)
            up2 = BatchNormalization()(up2)
            up2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(up2)
            merge2 = concatenate([up2, encode2], axis=-1)
            
            up1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(merge2)
            up1 = BatchNormalization()(up1)
            up1 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same')(up1)
            merge1 = concatenate([up1, encode1], axis=-1)

            decode0 = Conv2D(filters=32, kernel_size=3, padding='same')(merge1)
            decode0 = BatchNormalization()(decode0)

            decode0 = Conv2D(filters=label_num, kernel_size=3, padding='same')(decode0)
            decode0 = BatchNormalization()(decode0)

            output = Conv2D(filters=label_num, kernel_size=1, activation='softmax', padding='same')(decode0)

            model = Model(inputs=[input_tensor], outputs=output)

        if gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        if pre_train_model != None:
                print('load pritrain model weights')
                model.load_weights(pre_train_model)

        return model

    def make_ternausNet16_rgbd(self, label_num, is_trainable=True, pre_train_model=None, gpus=1):
        print('make model: ternausNet16')
        with tf.device("/cpu:0"):
            input_rgb = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
            input_depth = Input(shape=(DEPTH_SIZE, DEPTH_SIZE, 1))
            model_vgg = vgg16.VGG16(input_tensor=input_rgb, weights='imagenet', include_top=False)

            if is_trainable == False:
                for layer in model_vgg.layers[:15]:
                    layer.trainable = False

            layer_vgg = dict([(layer.name, layer.output) for layer in model_vgg.layers])

            encode1 = layer_vgg['block1_conv2']
            encode2 = layer_vgg['block2_conv2']
            encode3 = layer_vgg['block3_conv3']
            encode4 = layer_vgg['block4_conv3']
            encode5 = layer_vgg['block5_conv3']
            pool5 = layer_vgg['block5_pool']

            encode6 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(pool5)
            encode6 = BatchNormalization()(encode6)
            encode6 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same')(encode6)
            encode6 = BatchNormalization()(encode6)

            encode1_depth = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(input_depth)
            encode1_depth = BatchNormalization()(encode1_depth)
            encode1_depth = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(encode1_depth)
            encode1_depth = BatchNormalization()(encode1_depth)
            pool1_depth = MaxPool2D(pool_size=2, strides=2)(encode1_depth)

            #encode2_depth = Conv2D(filters=128, kernel_size=3, activation='relu',
            #padding='same')(pool1_depth)
            #encode2_depth = BatchNormalization()(encode2_depth)
            #encode2_depth = Conv2D(filters=128, kernel_size=3, activation='relu',
            #padding='same')(encode2_depth)
            #encode2_depth = BatchNormalization()(encode2_depth)
            #pool2_depth = MaxPool2D(pool_size=2, strides=2)(encode2_depth)

            #encode3_depth = Conv2D(filters=256, kernel_size=3, activation='relu',
            #padding='same')(pool2_depth)
            #encode3_depth = BatchNormalization()(encode3_depth)
            #encode3_depth = Conv2D(filters=256, kernel_size=3, activation='relu',
            #padding='same')(encode3_depth)
            #encode3_depth = BatchNormalization()(encode3_depth)
            #encode3_depth = Conv2D(filters=256, kernel_size=3, activation='relu',
            #padding='same')(encode3_depth)
            #encode3_depth = BatchNormalization()(encode3_depth)
            #pool3_depth = MaxPool2D(pool_size=2, strides=2)(encode3_depth)

            #encode4_depth = Conv2D(filters=512, kernel_size=3, activation='relu',
            #padding='same')(pool3_depth)
            #encode4_depth = BatchNormalization()(encode4_depth)
            #encode4_depth = Conv2D(filters=512, kernel_size=3, activation='relu',
            #padding='same')(encode4_depth)
            #encode4_depth = BatchNormalization()(encode4_depth)
            #encode4_depth = Conv2D(filters=512, kernel_size=3, activation='relu',
            #padding='same')(encode4_depth)
            #encode4_depth = BatchNormalization()(encode4_depth)
            #pool4_depth = MaxPool2D(pool_size=2, strides=2)(encode4_depth)

            depth_output = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(pool1_depth)
            depth_output = BatchNormalization()(depth_output)
            depth_output = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(depth_output)
            depth_output = BatchNormalization()(depth_output)

            # depth size must be 32 this time

            up5 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(encode6)
            up5 = BatchNormalization()(up5)
            up5 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(up5)
            merge5 = concatenate([up5, encode5], axis=-1)
            merge5 = concatenate([merge5, depth_output], axis=-1)

            up4 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(merge5)
            up4 = BatchNormalization()(up4)
            up4 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(up4)
            merge4 = concatenate([up4, encode4], axis=-1)
            
            up3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(merge4)
            up3 = BatchNormalization()(up3)
            up3 = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')(up3)
            merge3 = concatenate([up3, encode3], axis=-1)
            
            up2 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(merge3)
            up2 = BatchNormalization()(up2)
            up2 = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(up2)
            merge2 = concatenate([up2, encode2], axis=-1)
            
            up1 = Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(merge2)
            up1 = BatchNormalization()(up1)
            up1 = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same')(up1)
            merge1 = concatenate([up1, encode1], axis=-1)

            decode0 = Conv2D(filters=32, kernel_size=3, padding='same')(merge1)
            decode0 = BatchNormalization()(decode0)

            output = Conv2D(filters=label_num, kernel_size=1, activation='softmax', padding='same')(decode0)

            model = Model(inputs=[input_rgb, input_depth], outputs=output)

        if gpus > 1:
            model = multi_gpu_model(model, gpus=gpus)

        if pre_train_model != None:
                print('load pritrain model weights')
                model.load_weights(pre_train_model)

        return model

    def fit(self, model, dataset, scene_index='1', data_type='rgb', depth_type='estimated_depth'):

        adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['acc'])

        print(model.summary())

        if data_type=="rgb":
            histry = model.fit([dataset.train_image], dataset.train_label,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[ModelCheckpoint(filepath="./weights/"+data_type+"_"+scene_index+"_bestmodel.hdf5", monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True, period=1)], shuffle=True, validation_data=([dataset.val_image], dataset.val_label), verbose=1)
            self.plot_history(histry, scene_index, data_type)
        elif data_type=="rgbd":
            histry = model.fit([dataset.train_image, dataset.train_depth], dataset.train_label,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[ModelCheckpoint(filepath="./weights/"+data_type+"_"+depth_type+"_"+scene_index+"_bestmodel.hdf5", monitor='val_loss', save_best_only=True, mode='min', save_weights_only=True, period=1)], shuffle=True, validation_data=([dataset.val_image, dataset.val_depth], dataset.val_label), verbose=1)
            self.plot_history(histry, scene_index, data_type, depth_type)


        return histry

    def plot_history(self, history, scene_index, data_type, depth_type=""):
        if depth_type != "":
            depth_type= "_" + depth_type
        plt.rcParams["font.size"] = 12
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        # plt.title('accuracy')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('accuracy', fontsize=14)
        plt.legend(['train accuracy', 'validation accuracy'], loc='lower right')
        plt.savefig('acc_'+scene_index+'_'+data_type+depth_type+'.pdf')
        plt.figure()
        
        plt.rcParams["font.size"] = 12
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        # plt.title('loss')
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.legend(['train loss', 'validation loss'], loc='upper right')
        plt.savefig('loss_'+scene_index+'_'+data_type+depth_type+'.pdf')

def load_images(inputpath, imagesize, type_color, preprocessing="None"):
    imglist = []
    filenames = []
    print('preprocessing: ' + preprocessing)

    for root, dirs, files in os.walk(inputpath):
        for fn in sorted(files):
            bn, ext = os.path.splitext(fn)
            if ext not in [".bmp", ".BMP", ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]:
                continue
  
            filename = os.path.join(root, fn)
            
            if type_color == 'Color':
                if preprocessing == "None":
                    testimage = cv2.imread(filename, cv2.IMREAD_COLOR)
                    height, width = testimage.shape[:2]
                    testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)
                    testimage = np.asarray(testimage, dtype=np.float64)
                    testimage = testimage[::-1]
                    imglist.append(testimage)
                elif preprocessing == "Imagenet":
                    testimage = load_img(filename, target_size=(imagesize, imagesize))
                    testimage = img_to_array(testimage)
                    testimage = vgg16.preprocess_input(testimage)
                    imglist.append(testimage)

            elif type_color == 'Gray':
                testimage = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                height, width = testimage.shape[:2]
                testimage = cv2.resize(testimage, (imagesize, imagesize), interpolation = cv2.INTER_AREA)
                testimage = np.asarray([testimage], dtype=np.float64)
                testimage = np.asarray(testimage, dtype=np.float64).reshape((1, imagesize, imagesize))
                testimage = testimage.transpose(1, 2, 0)
                imglist.append(testimage)
                
            filenames.append(filename)
    imgsdata = np.asarray(imglist, dtype=np.float32)
    print(imgsdata.shape)
  
    return imgsdata, filenames

def make_binary_label(label_list):
    binary_label_list = []
    for i in range(len(label_list)):
        background = np.abs(label_list[i] - 1)  # ラベルを反転．背景にラベル付け
        binary_label = np.concatenate([label_list[i], background], axis=-1)
        binary_label_list.append(binary_label)

    return np.asarray(binary_label_list, dtype=np.int32)

def save_images(savepath, filenamelist, imagelist):
    for i, fn in enumerate(filenamelist):
        filename = os.path.join(savepath, fn)
        testimage = imagelist[i]
        cv2.imwrite(filename, testimage)