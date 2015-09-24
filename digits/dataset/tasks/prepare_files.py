# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import re

import digits
from digits import utils
from digits.utils import subclass, override
from digits.task import Task

# NOTE: Increment this everytime the pickled object
PICKLE_VERSION = 1

@subclass
class ClearFiles(Task):
    def __init__(self, job_dir, tmp_folder, **kwargs):
        self.tmp_folder = tmp_folder

        super(ClearFiles, self).__init__(job_dir, **kwargs)
        self.pickver_task_parsefolder = PICKLE_VERSION

    def __getstate__(self):
        state = super(ClearFiles, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(ClearFiles, self).__setstate__(state)

    @override
    def name(self):
        return "Clear tmp files"

    @override
    def html_id(self):
        return super(ClearFiles, self).html_id()

    @override
    def offer_resources(self, resources):
        key = 'create_db_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources):
        args = ["rm", '-rf',
                self.tmp_folder,
                "{}/{}".format(self.job_dir, "train.txt.tmp"),
                "{}/{}".format(self.job_dir, "val.txt.tmp"),
                "{}/{}".format(self.job_dir, "test.txt.tmp"),
                ]

        print " ".join(args)
        return args

    @override
    def process_output(self, line):
        return True

##################################################################

@subclass
class PrepareFiles(Task):
    def __init__(self, job_dir, output_file, input_file, resize_mode, mean_file, image_dims, encoding, **kwargs):
        self.output_file = output_file
        self.input_file = input_file
        self.resize_mode = resize_mode
        self.mean_file = mean_file
        self.image_dims = image_dims
        self.encoding = encoding

        super(PrepareFiles, self).__init__(job_dir)
        self.pickver_task_parsefolder = PICKLE_VERSION

    def __getstate__(self):
        state = super(PrepareFiles, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(PrepareFiles, self).__setstate__(state)

    @override
    def name(self):
        return "Prepare regression file"

    @override
    def html_id(self):
        return super(PrepareFiles, self).html_id()

    @override
    def offer_resources(self, resources):
        key = 'create_db_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources):
        args = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'prepare_regression_files.py'),
                self.output_file,
                self.input_file,
                self.resize_mode,
                self.mean_file,
                str(self.image_dims[0]),
                str(self.image_dims[1]),
                self.encoding
                ]

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Process (\d+)/(\d+)', message)
        if match:
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

        if level == 'warning':
            self.logger.warning('%s: %s' % (self.name(), message))
            return True
        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True
