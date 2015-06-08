# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from digits.job import Job
from . import tasks

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class EvaluationJob(Job):
    """
    A Job that creates a dataset
    """

    def __init__(self, **kwargs):
        """
        """
        super(EvaluationJob, self).__init__(**kwargs)
        self.pickver_job_evaluation = PICKLE_VERSION
 


    def accuracy_tasks(self):
        """Return all the Accuracy Tasks for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.AccuracyTask)]
