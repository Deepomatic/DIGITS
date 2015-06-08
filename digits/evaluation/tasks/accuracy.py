# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import caffe
import time
import math
import subprocess
from collections import Counter 

import pandas as pd
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
    """Compute the full accuracy"""
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
 
            argmax_probs = np.argmax(probas, axis=1)

            labels_masked = np.ma.compressed(np.ma.masked_array(self.labels_data, mask_seuil.mask))
            predict_masked = np.ma.compressed(np.ma.masked_array(self.prediction_data, mask_seuil.mask))

            N_threshold = predict_masked.shape[0]/float(N)
            acc = np.mean(labels_masked==predict_masked)
            return acc, N_threshold

        # return 100-200 values or fewer
        t = ['Threshold']
        accuracy = ['Accuracy']
        response = ['Recall']

        max_proba = np.max(self.probas_data)


        for i in range(20): 
            acc, num = f_seuil(max_proba * i / 20.0, self.probas_data)
            t += [max_proba * i / 20.0]
            accuracy += [acc] 
            response += [num]

        return  {        
            "x": "Threshold",
            "columns": [ t, accuracy, response ],
            "axes": {
                'Recall': 'y2' 
            } 
        }


    def confusion_matrix_data(self):
        """
        Returns the matrix confusion datas formatted for a C3.js graph
        """
        if self.probas_data == None:
            return None 
        # return 100-200 values or fewer
        train_task = self.job.model_job.train_task()
        dataset_train_task = train_task.dataset.train_db_task()
        dataset_labels = train_task.dataset.labels_file
        labels_str = pd.read_csv(dataset_train_task.path(dataset_labels),header=None,sep="", engine='python')[0]

        def accuracy_per_class(class_index):
            label_flat = self.labels_data.tolist()           
            try: 
                start = label_flat.index(class_index)
                stop = (len(label_flat) - 1) - label_flat[::-1].index(class_index) 
                return np.mean(self.prediction_data[start:stop+1]==self.labels_data[start:stop+1])
            except:
                return None

        def most_represented_class_per_class(class_index):
            label_flat = self.labels_data.tolist()
            try: 
                start = label_flat.index(class_index)
                stop = (len(label_flat) - 1) - label_flat[::-1].index(class_index) 
                c = Counter(self.prediction_data[start:stop+1])
                return map(lambda x:(labels_str[x[0]], x[1]/float(stop-start+1)),c.most_common())
            except: 
                return None

        results = ""
        for i in range(0,len(labels_str)): 
            acc = accuracy_per_class(i)
            if acc != None:
                results += "{0} - {1}%\n".format(labels_str[i], round(accuracy_per_class(i) * 100,2))
                classes = most_represented_class_per_class(i)
                for k in classes[0:10]:
                    results+= "\t{1}%\t -\t {0}\n".format(k[0], round(k[1]*100, 2))


        return  { "results" : results }

