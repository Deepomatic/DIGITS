# Copyright (c) 2014-2015, Deepomatic SAS  All rights reserved.

from ..job import EvaluationJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageEvaluationJob(EvaluationJob):
    """
    A Job that creates an image dataset
    """

    def __init__(self, **kwargs):
        """
        Arguments:
        """

        super(ImageEvaluationJob, self).__init__(**kwargs)
        self.pickver_job_evaluation_image = PICKLE_VERSION

