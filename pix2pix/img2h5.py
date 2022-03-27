# python img2h5.py -trp ./images/1/train -vap ./images/1/val -tep ./images/1/test -is 128
import numpy as np
import glob
import argparse
import h5py

from keras.preprocessing.image import load_img, img_to_array
import cv2

def make_img_dataset(input_paths, target_paths, image_size):
    input_imgs = []
    target_imgs = []
    for i in range(len(input_paths)):
        test_rgb_imgs = load_img(input_paths[i])
        test_depth_imgs = load_img(target_paths[i], color_mode="grayscale")

        input_imgarray = img_to_array(test_rgb_imgs)
        target_imgarray = img_to_array(test_depth_imgs)

        input_imgarray = cv2.resize(input_imgarray, (image_size, image_size))
        target_imgarray = cv2.resize(target_imgarray, (image_size, image_size))

        target_imgarray = np.reshape(target_imgarray, (image_size, image_size, 1))

        input_imgs.append(input_imgarray)
        target_imgs.append(target_imgarray)

    return np.array(input_imgs), np.array(target_imgs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', '-trp', required=True)
    parser.add_argument('--val_path', '-vap', required=True)
    parser.add_argument('--test_path', '-tep', required=True)
    parser.add_argument('--image_size', '-is', required=True)
    args = parser.parse_args()

    train_rgb_images_path = glob.glob(args.train_path+'/rgb/*')
    train_depth_images_path = glob.glob(args.train_path+'/depth/*')
    train_rgb_images_path.sort()
    train_depth_images_path.sort()

    val_rgb_images_path = glob.glob(args.val_path+'/rgb/*')
    val_depth_images_path = glob.glob(args.val_path+'/depth/*')
    val_rgb_images_path.sort()
    val_depth_images_path.sort()

    test_rgb_images_path = glob.glob(args.test_path + '/rgb/*')
    test_depth_images_path = glob.glob(args.test_path + '/depth/*')
    test_rgb_images_path.sort()
    test_depth_images_path.sort()

    image_size =  int(args.image_size)

    train_rgb_imgs, train_depth_imgs = make_img_dataset(train_rgb_images_path, train_depth_images_path, image_size)
    val_rgb_imgs, val_depth_imgs = make_img_dataset(val_rgb_images_path, val_depth_images_path, image_size)
    test_rgb_imgs, test_depth_imgs = make_img_dataset(test_rgb_images_path, test_depth_images_path, image_size)

    print('shapes')
    print('raw imgs : ', train_rgb_imgs.shape)
    print('gen imgs : ', train_depth_imgs.shape)
    print('val raw  : ', val_rgb_imgs.shape)
    print('val gen  : ', val_depth_imgs.shape)
    print('test raw : ', test_rgb_imgs.shape)
    print('test gen : ', test_depth_imgs.shape)

    outh5 = h5py.File('datasetimages.hdf5', 'w')
    outh5.create_dataset('train_data_gen', data=train_depth_imgs)
    outh5.create_dataset('train_data_raw', data=train_rgb_imgs)
    outh5.create_dataset('val_data_gen', data=val_depth_imgs)
    outh5.create_dataset('val_data_raw', data=val_rgb_imgs)
    outh5.create_dataset('test_data_gen', data=test_depth_imgs)
    outh5.create_dataset('test_data_raw', data=test_rgb_imgs)
    outh5.flush()
    outh5.close()

if __name__=='__main__':
    main()
