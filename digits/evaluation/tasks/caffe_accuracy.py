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
    """Compute the full accuracy"""
    def __init__(self, job, snapshot, db_task, **kwargs):
        """
        Arguments:
        job -- the job
        snapshot -- the snapshot

        Keyword arguments:
        """

        super(CaffeAccuracyTask, self).__init__(**kwargs)
        self.pickver_task_averageaccuracy = PICKLE_VERSION

        self.job = job
        self.snapshot = snapshot
        self.probas_data = None
        self.db_task = db_task
 

    def __getstate__(self):
        state = super(CaffeAccuracyTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(CaffeAccuracyTask, self).__setstate__(state)

    @override
    def name(self): 
        return 'Compute performance on '+self.db_task.db_name

    @override
    def offer_resources(self, resources):
        key = 'compute_accuracy_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources):
        
        train_task = self.job.model_job.train_task()
        dataset_val_task = self.db_task
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
                dataset_val_task.path(dataset_val_file),
                # Resize mode
                train_task.dataset.resize_mode
            ]
 

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False


        # progress
        match = re.match(r'Progress: ([-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)', message)
        if match:
            self.progress = float(match.group(1))
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
            snapshot_file, snapshot_extension = os.path.splitext(self.snapshot)

            self.probas_data = joblib.load(snapshot_file + "-accuracy-proba.pkl")
            self.labels_data = joblib.load(snapshot_file + "-accuracy-labels.pkl")
            self.prediction_data = self.probas_data.argmax(axis=1)

            avg_accuracy = self.avg_accuracy_graph_data()
            confusion_matrix = self.confusion_matrix_data()
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'accuracy_data',
                        'avg_accuracy': avg_accuracy,
                        'confusion_matrix': confusion_matrix,
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

