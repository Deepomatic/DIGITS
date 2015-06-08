# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import io
import re
import json
import math
import tarfile
import zipfile

from google.protobuf import text_format
from flask import render_template, request, redirect, url_for, flash, make_response, abort, jsonify
from caffe.proto import caffe_pb2
import caffe.draw

import digits
from digits import utils
from digits.webapp import app, scheduler, autodoc
import images.views
import images as evaluation_images


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

# @app.route(NAMESPACE + 'summary', methods=['GET'])
# @autodoc('evaluations')
# def evaluation_summary():
#     """
#     Return a short HTML summary of an EvaluationJob
#     """
#     job_id = request.args.get('job_id', '')
#     if not job_id:
#         return 'No job_id in request!'

#     job = scheduler.get_job(job_id)

#     return render_template('evalutions/summary.html', dataset=job)



