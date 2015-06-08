# -*- coding: utf-8 -*-

from caffe.proto import caffe_pb2
from digits import utils
from digits.webapp import app, scheduler, autodoc
from flask import render_template, request, redirect, url_for, flash, make_response, abort, jsonify
from google.protobuf import text_format

import caffe.draw
import digits
import images as evaluation_images
import images.views
import io
import json
import math
import os
import re
import tarfile
import zipfile

NAMESPACE = '/evaluations/'

@app.route(NAMESPACE + '<job_id>', methods=['GET'])
@autodoc('evaluations')
def evaluations_show(job_id):
    """
    Show an EvaluationJob
    """
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, evaluation_images.ImageEvaluationJob):
        return evaluation_images.classification.views.show(job)
    else:
        abort(404)
 
