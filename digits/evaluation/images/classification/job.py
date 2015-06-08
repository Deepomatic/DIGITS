# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
# -*- coding: utf-8 -*-

import os.path

from digits.dataset import tasks
from digits.model import tasks
from digits import utils
from digits.utils import subclass, override
from ..job import ImageEvaluationJob
from digits.webapp import app, scheduler
# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class ImageClassificationEvaluationJob(ImageEvaluationJob):
    """
    A Job that creates an image dataset for a classification network
    """

    def __init__(self, modeljob_id, model_epoch=None, **kwargs):

        self.model_job = scheduler.get_job(modeljob_id)
        self.model_epoch = model_epoch

        super(ImageClassificationEvaluationJob, self).__init__(**kwargs)
        self.pickver_job_evaluation_image_classification = PICKLE_VERSION

        self.labels_file = None

        # We get the snapshot file
        task = self.model_job.train_task()

        snapshot_filename = None 
        if self.model_epoch == -1 and len(task.snapshots):
            self.model_epoch = task.snapshots[-1][1]
            snapshot_filename = task.snapshots[-1][0]
        else:
            for f, e in task.snapshots:
                if e == self.model_epoch:
                    snapshot_filename = f
                    break
        if not snapshot_filename:
            raise ValueError('Invalid epoch')

        self.snapshot_filename = snapshot_filename




    @override
    def job_type(self):
        return 'Image Classification Model Evaluation'
 
