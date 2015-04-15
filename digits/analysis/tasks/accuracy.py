# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import caffe
import time
import math
import subprocess

import numpy as np
from google.protobuf import text_format
from caffe.proto import caffe_pb2

from digits.task import Task
from digits.config import config_option
from digits.status import Status
from digits import utils, dataset
from digits.utils import subclass, override, constants
from digits.dataset import ImageClassificationDatasetJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class AccuracyTask(Task):
    """Computer the full accuracy"""
    def __init__(self, **kwargs):
        """
        Arguments:
        job -- the job
        snapshot -- the snapshot

        Keyword arguments:
        percent_test -- percent of images used in the test set
        """

       
        super(AccuracyTask, self).__init__(**kwargs) 

    def __getstate__(self):
        state = super(AccuracyTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(AccuracyTask, self).__setstate__(state)
 
    def avg_accuracy_graph_data(self):
        """
        Returns the average accuracy datas formatted for a C3.js graph
        """

        if self.probas_data == None:
            return None

        def f_seuil(threshold, probas):
            N = len(probas)

            max_probs = np.max(probas, axis=1)
            mask_seuil = np.ma.masked_where(max_probs<threshold, max_probs)
 
            N_threshold = float(sum(mask_seuil))/float(N)
            argmax_probs = np.argmax(probas, axis=1)

            labels_masked = np.ma.compressed(np.ma.masked_array(self.labels_data, mask_seuil.mask))
            predict_masked = np.ma.compressed(np.ma.masked_array(self.prediction_data, mask_seuil.mask))
            acc = np.mean(labels_masked==predict_masked)
            return acc, N_threshold

        # return 100-200 values or fewer
        t = ['threshold']
        accuracy = ['acc']

        max_proba = np.max(self.probas_data)


        for i in range(20): 
            acc, num = f_seuil(max_proba * i / 20.0, self.probas_data)
            t += [max_proba * i / 20.0]
            accuracy += [acc] 


        return {
                'columns': [t, accuracy],
                'xs': {
                    'acc': 'threshold'
                    },
                'names': {
                    'acc': 'Accuracy'
                    },
                }
 
