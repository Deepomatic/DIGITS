# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import requests
import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits.utils.forms import validate_required_iff
import os
import json

class ImageRegressionDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new ImageRegressionDatasetJob
    """

    # method = wtforms.SelectField(u'Dataset type',
    #     choices = [
    #         ('folder', 'Folder'),
    #         ('textfile', 'Textfiles'),
    #         ('upload', 'upload'),
    #         ('advanced', 'advanced')
    #         ],
    #     default='folder',
    #     )

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
                validators.AnyOf(['regression', 'upload', 'advanced'], message='The method you chose is not currently supported.')
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

    def validate_file_path(form, field):
        if not field.data:
            raise Validators.ValidationError('Please fill all the fields')
        elif not os.path.exists(field.data) or os.path.isdir(field.data):
                raise validators.ValidationError('File does not exist')
        else:
            return True

    def validate_folder_path(form, field):
        if not field.data:
            raise Validators.ValidationError('Please fill all the fields')
        elif not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise validators.ValidationError(message='Folder does not exist')
        elif not os.path.exists(field.data + "/dataset.json") or not os.path.isfile(field.data + "/dataset.json"):
            raise validators.ValidationError(message='The path {} should contain a file named dataset.json'.format(field.data))
        else:
            #TODO maybe move this code in another validator
            with open(field.data + "/dataset.json") as fd:
                content = json.loads(fd.read())
                if content.has_key('path') and content.has_key('data') and content.has_key('labels'):
                    for key in content["data"]:
                        if not content["data"][key].has_key('type'):
                            raise validators.ValidationError(message="missing field type")
                        if not content["data"][key]["type"] in ("img", "other", "box", "class"):
                            raise validators.ValidationError(message="type must be of type (img,other,box,class)")
                    for key in content["labels"]:
                        if not content["labels"][key].has_key('type'):
                            raise validators.ValidationError(message="missing field type")
                        if not content["labels"][key]["type"] in ("img", "other", "box", "class"):
                            raise validators.ValidationError(message="type must be of type (img,other,box,class)")
                else:
                    raise validators.ValidationError(message="missing field path, data or labels")
            return True

    textfile_folderPath = wtforms.StringField(u'Folder path',
    validators = [
        validate_required_iff(
            method='advanced'),
            validate_folder_path
        ]
    )

    ## file path
    textfile_filesPath = wtforms.StringField(u'Files file path',
            validators=[
                validate_required_iff(
                    method='upload'),
                    validate_file_path
                ]
            )

    textfile_labelsPath = wtforms.StringField(u'Labels file path',
            validators=[
                validate_required_iff(
                    method='upload'),
                    validate_file_path
                ]
            )
