# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import ImageModelForm

class ImageRegressionModelForm(ImageModelForm):
    """
    Defines the form used to create a new ImageRegressionModelJob
    """
    learning_rate = wtforms.FloatField('Base Learning Rate',
            default = 0.0001,
            validators = [
                validators.NumberRange(min=0),
                ]
            )

