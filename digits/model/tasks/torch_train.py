# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import time
import math
import subprocess

import numpy as np

import digits
from train import TrainTask
from digits.config import config_option
from digits.status import Status
from digits import utils, dataset
from digits.utils import subclass, override, constants
from digits.dataset import ImageClassificationDatasetJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class TorchTrainTask(TrainTask):
    """
    Trains a torch model
    """

    CAFFE_LOG = 'train.log'

    @staticmethod
    def upgrade_network(cls, network):
        #TODO
        pass

    def __init__(self, network, job_path, **kwargs):
        """
        Arguments:
        network -- a NetParameter defining the network
        """
        super(TorchTrainTask, self).__init__(**kwargs)
        self.pickver_task_caffe_train = PICKLE_VERSION

        self.current_iteration = 0
        self.last_pop = -1
        self.root = kwargs['dataset'].dir()
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.classifier = None
        self.network = network
        self.max_epoch = kwargs['train_epochs']

        self.solver_file = constants.CAFFE_SOLVER_FILE
        self.train_val_file = constants.CAFFE_TRAIN_VAL_FILE
        self.snapshot_prefix = constants.CAFFE_SNAPSHOT_PREFIX
        self.deploy_file = constants.CAFFE_DEPLOY_FILE
        self.args = kwargs
        self.path = job_path
        self.current_epoch = 1


        self.nbrImages  = kwargs['dataset'].tasks[0].train_count
        self.batchSize = kwargs['batch_size'] if isinstance(kwargs['batch_size'], int) else 128

        self.max_iter = self.nbrImages / self.batchSize * self.max_epoch



    def __getstate__(self):
        state = super(TorchTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'labels' in state:
            del state['labels']
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'caffe_log' in state:
            del state['caffe_log']

        return state

    def __setstate__(self, state):
        super(TorchTrainTask, self).__setstate__(state)

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None
        self.classifier = None


    def new_iteration(self):
        """
        Update current_iteration
        """
        self.current_iteration += 15 #lua test each 15 iter
        print "curr:{}, epoch:{}, max_iter:{}, progress:{}".format(self.current_iteration, self.current_epoch, self.max_iter, self.current_iteration / float(self.max_iter))
        self.send_progress_update(self.current_iteration / float(self.max_iter))

    ### Task overrides

    @override
    def task_arguments(self, **kwargs):
        gpu_id = kwargs.pop('gpu_id', None)
        args = ["th", os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'fbcunn', 'main.lua')]
        args += ["-data", self.root]
        args += ["-LR", str(self.args['learning_rate'])]
        args += ["-cache", self.path]
        args += ["-nGPU", "1"]
        args += ["-GPU", "3"]
        args += ["-backend", "cudnn"]
        args += ["-netType", "alexnet"] #mettre le nom du reseau
        args += ["-nEpochs", str(self.max_epoch)]
        args += ["-epochSize", str(int(self.nbrImages / self.batchSize))]
        args += ["-batchSize", str(self.batchSize)]
        args += ["-model", self.network]

        print " ".join(args)
        return args


    @override
    def name(self):
        return 'Train Torch Model'

    @override
    def before_run(self):
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None

    def read_labels(self):
        """
        Read labels from self.labels_file and store them at self.labels
        Returns True if at least one label was read
        """
        # TODO: move to TrainTask

        # The labels might be set already
        if hasattr(self, 'labels') and self.labels and len(self.labels) > 0:
            return True

        assert hasattr(self.dataset, 'labels_file'), 'labels_file not set'
        assert self.dataset.labels_file, 'labels_file not set'
        assert os.path.exists(self.dataset.path(self.dataset.labels_file)), 'labels_file does not exist'

        labels = []
        with open(self.dataset.path(self.dataset.labels_file)) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)

        assert len(labels) > 0, 'no labels in labels_file'

        self.labels = labels
        return True

    @override
    def process_output(self, line):
        match = re.search("(\S{10} \S{8}) Epoch (\d+) Accuracy top1-%: (\d+\.?\d+).*Loss: (\d+\.?\d+).*", str(line.strip()))
        print line, match
        if match:
            self.current_epoch = match.group(2)
            self.new_iteration() ## CHECK 
            accuracy = match.group(3)
            loss = match.group(4)
            self.save_train_output("accuracy", "Accuracy", float(accuracy)/100)
            self.save_train_output("loss", "Loss", float(loss))

            return True
        match = re.search(r"Epoch:.*\[(\d+)\]\[TESTING SUMMARY\] Total Time\(s\): \d+.\d+\s+\S+ \S+ \(per batch\): (\d+.\d+)\s+.*top-1 (\d+.\d+)", line)
        if match:
            self.current_epoch = match.group(1)
            self.new_iteration()
            accuracy = match.group(3)
            loss = match.group(2)
            self.save_val_output("accuracy", "Accuracy", float(accuracy)/100)
            self.save_val_output("loss", "Loss", float(loss))
            return True

        return True #hack to avoid flood in the output 

    def send_iteration_update(self, it):
        """
        Sends socketio message about the current iteration
        """
        # TODO: move to TrainTask
        from digits.webapp import socketio

        # if self.current_iteration == it:
        #     return

        self.current_iteration = it
        self.progress = float(it)/self.max_iter
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

    def send_data_update(self, important=False):
        """
        Send socketio updates with the latest graph data

        Keyword arguments:
        important -- if False, only send this update if the last unimportant update was sent more than 5 seconds ago
        """
        # TODO: move to TrainTask
        from digits.webapp import socketio

        if not important:
            if self.last_unimportant_update and (time.time() - self.last_unimportant_update) < 5:
                return
            self.last_unimportant_update = time.time()

        # loss graph data
        data = self.loss_graph_data()
        if data:
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'loss_graph',
                        'data': data,
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

        # lr graph data
        data = self.lr_graph_data()
        if data:
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'lr_graph',
                        'data': data,
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

    def send_snapshot_update(self):
        """
        Sends socketio message about the snapshot list
        """
        # TODO: move to TrainTask
        from digits.webapp import socketio

        socketio.emit('task update',
                {
                    'task': self.html_id(),
                    'update': 'snapshots',
                    'data': self.snapshot_list(),
                    },
                namespace='/jobs',
                room=self.job_id,
                )

    ### TrainTask overrides

    @override
    def detect_snapshots(self):
        pass
        # TODO

    @override
    def est_next_snapshot(self):
        # TODO: move to TrainTask
        if self.status != Status.RUN or self.current_iteration == 0:
            return None
        elapsed = time.time() - self.status_updates[-1][1]
        next_snapshot_iteration = (1 + self.current_iteration//self.snapshot_interval) * self.snapshot_interval
        return (next_snapshot_iteration - self.current_iteration) * elapsed // self.current_iteration

    @override
    def can_view_weights(self):
        return False

    @override
    def can_infer_one(self):
        return False

    @override
    def can_infer_many(self):
        return False

