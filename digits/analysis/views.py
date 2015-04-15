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
from digits.webapp import app, scheduler
import images.views
import images as analysis_images


NAMESPACE = '/analysis/'


@app.route(NAMESPACE + '<job_id>', methods=['GET'])
def analysis_show(job_id):
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, analysis_images.ImageAnalysisJob):
        return analysis_images.classification.views.show(job)
    else:
        abort(404)