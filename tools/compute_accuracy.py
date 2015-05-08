#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015, DEEPOMATIC SAS,  All rights reserved.

import sys
import os
import re
import argparse
import time
import logging
from random import shuffle
from urlparse import urlparse
import urllib

import pandas as pd
import numpy as np
import joblib, skimage


caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')


import caffe
import customClassifier

import requests

# Add path for DIGITS package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from digits import utils
import digits.log

logger = logging.getLogger('digits.tools.parse_folder')

def unescape(s):
    return urllib.unquote(s) 

def compute_accuracy(snapshot, deploy_file, labels_file, mean_file, val_file): 
    """
    Parses a folder of images into three textfiles
    Returns True on sucess

    Arguments:
    folder -- a folder containing folders of images (can be a filesystem path or a url)
    labels_file -- file for labels

    Keyword Arguments: 
    """

    # Read validation datas
    val_matrix = pd.read_csv(val_file, engine='python', header=None,sep=r" (?!(.*) )")

    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_data = open(mean_file, 'rb').read()
    mean_blob.ParseFromString(mean_data)
    mean_arr = np.array(caffe.io.blobproto_to_array(mean_blob))[0]

    # Loading the classifier
    caffe.set_mode_gpu()
    net = customClassifier.Classifier(deploy_file, snapshot,
                       mean=mean_arr.mean(1).mean(1),
                       channel_swap=(2,1,0),
                       raw_scale=255)

    size = len(val_matrix)
    num_classes = np.array(val_matrix[2])[-1]+1
    prediction = np.zeros(size, dtype=int)
    labels = np.zeros(size, dtype=int)
    probas = np.zeros([size, num_classes], dtype=float)
 
    for i in range(0, size): 
        cur_image = val_matrix[0][i]
        input_image = utils.image.load_image(cur_image)
        input_image = utils.image.resize_image(input_image, 256, 256, resize_mode='half_crop')
        input_image = skimage.img_as_float(input_image).astype(np.float32)

        labels[i] = val_matrix[2][i]
        probas[i] = net.predict([input_image], oversample=False)
        prediction[i] = probas[i].argmax()
        if i % (size/500) == 0:
            logger.debug("Progress: %0.2f" % (i/float(size)))

    snapshot_file, snapshot_extension = os.path.splitext(snapshot)


    joblib.dump(probas, snapshot_file + "-accuracy-proba.pkl")
    joblib.dump(labels, snapshot_file + "-accuracy-labels.pkl")
    logger.debug("Done")

  
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accuracy-Computing tool - DIGITS')

    ### Positional arguments

    # parser.add_argument('job_dir',
    #         help='A filesystem path to the job folder'
    #         )
    parser.add_argument('snapshot',
            help='The file containing the model snapshot.'
            )

    parser.add_argument('deploy_file',
            help='The file containing the deploy_file.'
            )

    parser.add_argument('labels',
            help='The file containing the labels.'
            )

    parser.add_argument('mean_file',
            help='The file containing the mean file.'
            )
    parser.add_argument('val_file',
            help='The file containing the val.txt'
            )
 
    ### Optional arguments

    # parser.add_argument('-t', '--train_file',
    #         help='The output file for training images'
    #         ) 

    args = vars(parser.parse_args())

    # for valid in [
    #         validate_folder(args['folder']),
    #         validate_range(args['percent_train'], min=0, max=100, allow_none=True),
    #         validate_output_file(args['train_file']),
    #         validate_range(args['percent_val'], min=0, max=100, allow_none=True),
    #         validate_output_file(args['val_file']),
    #         validate_range(args['percent_test'], min=0, max=100, allow_none=True),
    #         validate_output_file(args['test_file']),
    #         validate_range(args['min'], min=1),
    #         validate_range(args['max'], min=1, allow_none=True),
    #         ]:
    #     if not valid:
    #         sys.exit(1)

    start_time = time.time()

    if compute_accuracy(args['snapshot'], args['deploy_file'] ,args['labels'] ,args['mean_file'], args['val_file']):# ,args['val_file']):
        logger.info('Done after %d seconds.' % (time.time() - start_time))
        sys.exit(0)
    else:
        sys.exit(1)

