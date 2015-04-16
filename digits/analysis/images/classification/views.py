# Copyright (c) 2014-2015, DEEPOMATIC SAS.  All rights reserved.
# -*- coding: utf-8 -*-

import os
import re
import sys
import shutil
import tempfile
import random

import numpy as np
from flask import render_template, request, redirect, url_for, flash
from google.protobuf import text_format
from caffe.proto import caffe_pb2

import digits
from digits.config import config_option
from digits import utils
from digits.webapp import app, scheduler
from digits.analysis import tasks
from job import ImageClassificationAnalysisJob
from digits.status import Status

NAMESPACE = '/analysis/images/classification'


@app.route(NAMESPACE + '/new', methods=['GET'])
def image_classification_analysis_new():
    # form = ImageClassificationAnalysisForm()
    return render_template('analysis/images/classification/new.html', form=form)



@app.route(NAMESPACE + '/average_accuracy', methods=['POST'])
def image_classification_analysis_create():
    """Display the average_accuracy of a model """

    modelJob = scheduler.get_job(request.args['job_id'])
 
    if not modelJob:
        return 'Unknown model job_id "%s"' % request.args['job_id'], 500

    job = None
    try:
        # We retrieve the selected snapshot
        epoch = None
        if 'snapshot_epoch' in request.form:
            epoch = int(request.form['snapshot_epoch'])


        job = ImageClassificationAnalysisJob( 
            name=modelJob._name+"-accuracy-analysis",
            modeljob_id= modelJob.id(),
            model_epoch= epoch
            )
        
        job.tasks.append(
                tasks.CaffeAccuracyTask(                    
                    job_dir         = job.dir(),
                    job             = job,
                    snapshot        = job.snapshot_filename,
                    percent_test    = 10,
                    )
                )

        scheduler.add_job(job)
        return redirect(url_for('analysis_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

 
def show(job):
    """
    Called from digits.analysis.views.analysis_show()
    """
    return render_template('analysis/images/classification/show.html', job=job)
