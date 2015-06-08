#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import joblib, skimage
import logging
import numpy as np
import os
import pandas as pd
import sys
import time

# Add path for DIGITS package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log
import caffe

logger = logging.getLogger('digits.tools.compute_accuracy')


class Classifier(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    It differs from Caffe original class version by 
    taking the last output layer instead of the first (which
    is not always the right output layer, e.g in the case
    of googlenet)
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        """
        Take
        image_dims: dimensions to scale input for cropping/sampling.
            Default is to scale to net input size for whole-image crop.
            mean, input_scale, raw_scale, channel_swap: params for
            preprocessing options.
        """
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        # configure pre-processing
        in_ = self.inputs[0]
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2,0,1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims


    def predict(self, inputs, out_layer_name=None, oversample=True):
        """
        Predict classification probabilities of inputs.

        Take
        inputs: iterable of (H x W x K) input ndarrays.
        oversample: average predictions across center, corners, and mirrors
                    when True (default). Center-only prediction when False.

        Give
        predictions: (N x C) ndarray of class probabilities
                     for N images and C classes.
        """
        # Scale to standardize input dimensions.
        input_ = np.zeros((len(inputs),
            self.image_dims[0], self.image_dims[1], inputs[0].shape[2]),
            dtype=np.float32)
        for ix, in_ in enumerate(inputs):
            input_[ix] = caffe.io.resize_image(in_, self.image_dims)

        if oversample:
            # Generate center, corner, and mirrored crops.
            input_ = caffe.io.oversample(input_, self.crop_dims)
        else:
            # Take center crop.
            center = np.array(self.image_dims) / 2.0
            crop = np.tile(center, (1, 2))[0] + np.concatenate([
                -self.crop_dims / 2.0,
                self.crop_dims / 2.0
            ])
            input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

        # Classify
        caffe_in = np.zeros(np.array(input_.shape)[[0,3,1,2]],
                            dtype=np.float32)
        for ix, in_ in enumerate(input_):
            caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
        out = self.forward_all(**{self.inputs[0]: caffe_in})

        # Where it differs from the original Caffe class
        if out_layer_name == None:
            predictions = out[self.outputs[-1]]
        else:
            predictions = out[out_layer_name]

        # For oversampling, average predictions across crops.
        if oversample:
            predictions = predictions.reshape((len(predictions) / 10, 10, -1))
            predictions = predictions.mean(1)

        return predictions
 

def compute_accuracy(snapshot, deploy_file, labels_file, mean_file, img_set, width, height, resize_mode, oversample=False): 
    """
    Evaluate a Net on a set of images, and dump the result in two
    pickle files. 
    Returns True on sucess

    Arguments:
    snapshot -- a Caffe trained model 
    deploy_file -- the corresponding deploy file
    labels_file -- the file containing the dataset labels
    mean_file -- the dataset mean file
    img_set -- the file containing a list of images 
    resize_mode -- the mode used to resize images

    Keyword arguments:
    oversampling -- boolean, True if oversampling should be enabled
    """

    img_matrix = pd.read_csv(img_set, engine='python', header=None,sep=r" (?!(.*) )")

    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_data = open(mean_file, 'rb').read()
    mean_blob.ParseFromString(mean_data)
    mean = np.array(caffe.io.blobproto_to_array(mean_blob))[0].mean(1).mean(1)

    # Loading the classifier
    caffe.set_mode_gpu()
    net = Classifier(deploy_file, snapshot,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255)

    size = len(img_matrix)
    num_classes = np.array(img_matrix[2])[-1]+1
    prediction = np.zeros(size, dtype=int)
    labels = np.zeros(size, dtype=int)
    probas = np.zeros([size, num_classes], dtype=float)
 
    for i in range(0, size): 
        cur_image = img_matrix[0][i]
        input_image = utils.image.load_image(cur_image)
        input_image = utils.image.resize_image(input_image, width, height, resize_mode=resize_mode)
        input_image = skimage.img_as_float(input_image).astype(np.float32)

        labels[i] = img_matrix[2][i]
        probas[i] = net.predict([input_image], oversample=False)
        prediction[i] = probas[i].argmax()
        if size > 500 and i % (size/500) == 0:
            logger.debug("Progress: %0.2f" % (i/float(size)))

    snapshot_file, snapshot_extension = os.path.splitext(snapshot)

    # Dumping the result
    joblib.dump(probas, snapshot_file + "-accuracy-proba.pkl")
    joblib.dump(labels, snapshot_file + "-accuracy-labels.pkl")
    logger.debug("Done")
  
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Accuracy-Computing tool - DIGITS')
 
    parser.add_argument('snapshot',
            help='The Caffe model snapshot.'
            )

    parser.add_argument('deploy_file',
            help='The deploy_file.'
            )

    parser.add_argument('labels',
            help='The file containing the labels.'
            )

    parser.add_argument('mean_file',
            help='The dataset mean file.'
            )
    parser.add_argument('img_set',
            help='The file representing the set of files to evaluate'
            )
 
    parser.add_argument('width',            
            type=int,
            help='images width'
            )
    parser.add_argument('height',         
            type=int,
            help='images height'
            )
    parser.add_argument('resize_mode',
            help='The mode used to resize images'
            )


    args = vars(parser.parse_args())
    start_time = time.time()

    if compute_accuracy(
        args['snapshot'], 
        args['deploy_file'] ,
        args['labels'] ,
        args['mean_file'], 
        args['img_set'], 
        args['width'],
        args['height'],
        args['resize_mode']):
        logger.info('Done after %d seconds.' % (time.time() - start_time))
        sys.exit(0)
    else:
        sys.exit(1)

