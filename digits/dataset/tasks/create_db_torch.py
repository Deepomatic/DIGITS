# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import re
import operator

import digits
from digits import utils
from digits.utils import subclass, override
from digits.status import Status
from digits.task import Task

# NOTE: Increment this everytime the pickled version changes
PICKLE_VERSION = 1

@subclass
class CreateDbTorsh(Task):
    """Creates a database"""

    def __init__(self, input_file, db_name, image_dims, **kwargs):
        """
        Arguments:
        input_file -- read images and labels from this file
        db_name -- save database to this location
        image_dims -- (height, width, channels)

        Keyword Arguments:
        image_folder -- prepend image paths with this folder
        resize_mode -- used in utils.image.resize_image()
        encode -- save encoded JPEGs
        mean_file -- save mean file to this location
        backend -- type of database to use
        labels_file -- used to print category distribution
        """
        # Take keyword arguments out of kwargs
        self.image_folder = kwargs.pop('image_folder', None)
        self.resize_mode = kwargs.pop('resize_mode' , None)
        self.encode = kwargs.pop('encode', True)
        self.mean_file = kwargs.pop('mean_file', None)
        self.backend = kwargs.pop('backend', None)
        self.labels_file = kwargs.pop('labels_file', None)

        super(CreateDbTorsh, self).__init__(**kwargs)
        self.pickver_task_createdb = PICKLE_VERSION

        self.input_file = input_file
        self.db_name = db_name
        self.image_dims = image_dims
        self.path = kwargs.pop('job_dir', True)

        self.entries_count = None
        self.distribution = None

    def __getstate__(self):
        d = super(CreateDbTorsh, self).__getstate__()
        if 'labels' in d:
            del d['labels']
        return d

    def __setstate__(self, state):
        super(CreateDbTorsh, self).__setstate__(state)

    @override
    def name(self):
        #if self.db_name == utils.constants.TRAIN_DB or 'train' in self.db_name.lower():
        return 'Create DB'
        # elif self.db_name == utils.constants.VAL_DB or 'val' in self.db_name.lower():
        #     return 'Create DB (val)'
        # elif self.db_name == utils.constants.TEST_DB or 'test' in self.db_name.lower():
        #     return 'Create DB (test)'
        # else:
        #     return 'Create DB (%s)' % self.db_name

    @override
    def html_id(self):
        #if self.db_name == utils.constants.TRAIN_DB or 'train' in self.db_name.lower():
        return 'task-create_db-train'
        # elif self.db_name == utils.constants.VAL_DB or 'val' in self.db_name.lower():
        #     return 'task-create_db-val'
        # elif self.db_name == utils.constants.TEST_DB or 'test' in self.db_name.lower():
        #     return 'task-create_db-test'
        # else:
        #     return super(CreateDbTask, self).html_id()

    @override
    def task_arguments(self, **kwargs):
        args = ["th", os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'fbcunn', 'generate_db.lua')]
        args += "-data {0} -LR 0.01 -cache {1} -name {2} -size {3}".format(self.path, self.path, self.db_name, self.image_dims[0]).split()
        print args
        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        print line
        match = re.search(r'estimation:\D+(\d+)\D+(\d+)', line)
        if match and float(match.group(2)) > 0:
            self.progress = float(match.group(1))/float(match.group(2))
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


        return True

    def get_labels(self):
        """
        Read labels from labels_file and return them in a list
        """
        pass


    def distribution_data(self):
        """
        Returns distribution data for a C3.js graph
        """
        pass

