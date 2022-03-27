import os
import argparse
import numpy as np
import tensorflow as tf
from models import Network, Dataset

#set of random seed
#import random as rn
#import tensorflow as tf
#os.environ['PYTHONHASHSEED'] = '7'
#np.random.seed(7)
#rn.seed(7)
#session_conf = tf.ConfigProto(
#    intra_op_parallelism_threads=1,
#    inter_op_parallelism_threads=1
#)
#from keras import backend as K
#tf.set_random_seed(7)
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Instruments Segmentation')
    parser.add_argument('--data_type', '-d', type=str, default='rgb')
    parser.add_argument('--case', '-c', type=str, default="1")
    parser.add_argument('--depth_type', '-depth', type=str, default='estimated_depth')
    parser.add_argument('--gpus', '-g', type=int, default=1)
    args = parser.parse_args()
    
    if args.data_type == 'rgb':
        net = Network()
        model = net.make_ternausNet16_paper(label_num=2, is_trainable=True, gpus=args.gpus)
        # model.summary()

    elif args.data_type=='rgbd':
        net = Network()
        model = net.make_ternausNet16_rgbd(label_num=2, is_trainable=True, gpus=args.gpus)
        # model.summary()

    # load dataset
    dataset = Dataset('Imagenet', args.case, args.data_type, args.depth_type)
    
    # training
    histry = net.fit(model=model, dataset=dataset, scene_index=args.case, data_type=args.data_type, depth_type=args.depth_type)

    del net
