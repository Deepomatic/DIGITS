# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import requests

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff

class ImageRegressionDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new ImageClassificationDatasetJob
    """

    def required_if_method(value):

        def _required(form, field):
            if form.method.data == value:
                if field.data is None or (isinstance(field.data, str) and not field.data.strip()) or (isinstance(field.data, FileStorage) and not field.data.filename.strip()):
                    raise validators.ValidationError('This field is required.')
            else:
                field.errors[:] = []
                raise validators.StopValidation()

        return _required

    def validate_percent_val(form, field):
        if int(form.percent_val.data) +  int(form.percent_test.data) > 99:
            raise validators.ValidationError('The sum of the percentage of the val and test data should be bellow 99')
        return True

    def validate_test_val(form, field):
        if int(form.percent_val.data) +  int(form.percent_test.data) > 99:
            raise validators.ValidationError('The sum of the percentage of the val and test data should be bellow 99')
        return True


    method = wtforms.HiddenField(u'Dataset type',
            default='regression',
            validators=[
                validators.AnyOf(['regression'], message='The method you chose is not currently supported.')
                ]
            )
    input_files = wtforms.FileField(u'Input file',
            validators=[
                required_if_method('textfile'),

                ]
            )

    input_labels = wtforms.FileField(u'Input labels', 
            validators=[
                required_if_method('textfile'),

                ]
        )


    percent_val = wtforms.IntegerField(u'% for validation',
            default=25,
            validators=[
                validators.NumberRange(min=0, max=50),
                ]
            )

    percent_test = wtforms.IntegerField(u'% for validation',
            default=0,
            validators=[
                validators.NumberRange(min=0, max=50),
                ]
            )


    textfile_shuffle = wtforms.BooleanField('Shuffle lines',
            default = True)