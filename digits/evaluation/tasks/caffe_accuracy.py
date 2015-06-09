# -*- coding: utf-8 -*-

from accuracy import AccuracyTask
from caffe.proto import caffe_pb2
from digits import utils
from digits.utils import subclass, override
import digits
import joblib
import os
import os.path
import re
import sys

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class CaffeAccuracyTask(AccuracyTask):
    """
    Computes the accuracy/recall and confusion matrix
    corresponding to a given snapshot of a train task
    """
    def __init__(self, job, snapshot, db_task, **kwargs):
        """
        Arguments:
        job -- the job
        snapshot -- the snapshot file of the trained net
        db_task -- the dataset db on which we evaluate the net

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
                # Img width
                train_task.dataset.image_dims[1],
                # Img height
                train_task.dataset.image_dims[0],
                # Resize mode
                train_task.dataset.resize_mode
            ]


        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        _, level, message = self.preprocess_output_digits(line)
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
            snapshot_file, _ = os.path.splitext(self.snapshot)

            self.probas_data = joblib.load(snapshot_file + "-accuracy-proba.pkl")
            self.labels_data = joblib.load(snapshot_file + "-accuracy-labels.pkl")
            self.prediction_data = self.probas_data.argmax(axis=1)

            # Send the accuracy/recall curve and confusion matrix datas
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

