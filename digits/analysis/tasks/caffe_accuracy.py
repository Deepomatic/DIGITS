# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
# -*- coding: utf-8 -*-

import sys
import os.path
import os
import re 
import caffe
import time
import math
import subprocess
import digits



import numpy as np
import joblib

from google.protobuf import text_format
from caffe.proto import caffe_pb2

from digits.task import Task
from accuracy import AccuracyTask
from digits.config import config_option
from digits.status import Status
from digits import utils, dataset
from digits.utils import subclass, override, constants

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class CaffeAccuracyTask(AccuracyTask):
    """Computer the full accuracy"""
    def __init__(self, job, snapshot, **kwargs):
        """
        Arguments:
        job -- the job
        snapshot -- the snapshot

        Keyword arguments:
        percent_test -- percent of images used in the test set
        """

        # Take keyword arguments out of kwargs
        percent_test = kwargs.pop('percent_test', None) 

        super(CaffeAccuracyTask, self).__init__(**kwargs)
        self.pickver_task_averageaccuracy = PICKLE_VERSION

        self.job = job
        self.snapshot = snapshot
        self.probas_data = None

        if percent_test is None:
            self.percent_test = 10
        else:
            pct = float(percent_test)
            if pct < 0:
                pct = 0
            elif pct > 100:
                raise ValueError('percent_test must not exceed 100')
            self.percent_test = pct
  

    def __getstate__(self):
        state = super(CaffeAccuracyTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(CaffeAccuracyTask, self).__setstate__(state)

    @override
    def name(self): 
        return 'Computer average accuracy for model (%s - %s)' % (self.job.dir(), self.snapshot)

    @override
    def task_arguments(self, **kwargs):
        
        train_task = self.job.model_job.train_task()
        dataset_val_task = train_task.dataset.val_db_task()
        dataset_train_task = train_task.dataset.train_db_task()

        deploy_file = train_task.deploy_file
        dataset_labels = train_task.dataset.labels_file
        dataset_mean_file = dataset_train_task.mean_file
        dataset_val_file = dataset_val_task.input_file

        args = [sys.executable, 
            os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'compute_accuracy.py'),
                # Caffe model path
                self.snapshot, 
                # Deploy file
                train_task.path(deploy_file),
                # Labels
                dataset_train_task.path(dataset_labels),
                # Mean file
                dataset_train_task.path(dataset_mean_file),
                # val.txt
                dataset_val_task.path(dataset_val_file)
                #self.path(utils.constants.LABELS_FILE)
                 ]

        # if (self.percent_val + self.percent_test) < 100:
        #     args.append('--train_file=%s' % self.path(utils.constants.TRAIN_FILE))
        # if self.percent_val > 0:
        #     args.append('--val_file=%s' % self.path(utils.constants.VAL_FILE))
        #     args.append('--percent_val=%s' % self.percent_val)
        # if self.percent_test > 0:
        #     args.append('--test_file=%s' % self.path(utils.constants.TEST_FILE))
        #     args.append('--percent_test=%s' % self.percent_test)
        # if self.max_per_category is not None:
        #     args.append('--max=%s' % self.max_per_category)

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        print message

        # progress
        match = re.match(r'Progress: ([-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)', message)
        if match:
            self.progress = float(match.group(1))
            print self.progress
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'progress',
                        'percentage': int(round(100*self.progress)),
                        'eta': utils.time_filters.print_time_diff(self.est_done()),
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )
            return True

        # totals
        match = re.match(r'Done', message)
        if match:
            # Store the accuracy data
            print "Done ! Loading datas..."
            snapshot_file, snapshot_extension = os.path.splitext(self.snapshot)

            self.probas_data = joblib.load(snapshot_file + "-accuracy-proba.pkl")
            self.labels_data = joblib.load(snapshot_file + "-accuracy-labels.pkl")
            print self.probas_data.shape
            self.prediction_data = self.probas_data.argmax(axis=1)
            print self.prediction_data.shape

            avg_accuracy = self.avg_accuracy_graph_data()
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'avg_accuracy',
                        'data': avg_accuracy
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

            self.logger.debug(self.probas_data.shape)
            return True
 

        if level == 'warning':
            self.logger.warning('%s: %s' % (self.name(), message))
            return True
        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True

