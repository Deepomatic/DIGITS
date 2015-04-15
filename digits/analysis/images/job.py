# Copyright (c) 2014-2015, Deepomatic SAS  All rights reserved.

from ..job import AnalysisJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageAnalysisJob(AnalysisJob):
    """
    A Job that creates an image dataset
    """

    def __init__(self, **kwargs):
        """
        Arguments:
        """

        super(ImageAnalysisJob, self).__init__(**kwargs)
        self.pickver_job_analysis_image = PICKLE_VERSION

