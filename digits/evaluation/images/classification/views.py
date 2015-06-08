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
from digits.evaluation import tasks
from job import ImageClassificationEvaluationJob
from digits.status import Status

NAMESPACE = '/evaluations/images/classification'


@app.route(NAMESPACE + '/new', methods=['GET'])
def image_classification_evaluation_new():
    # form = ImageClassificationEvaluationForm()
    return render_template('evaluations/images/classification/new.html', form=form)


@app.route(NAMESPACE + '/accuracy', methods=['POST'])
def image_classification_evaluation_create():
    """Display the accuracy of a model """

    modelJob = scheduler.get_job(request.args['job_id'])
 
    if not modelJob:
        return 'Unknown model job_id "%s"' % request.args['job_id'], 500

    job = None
    try:
        # We retrieve the selected snapshot
        epoch = None
        if 'snapshot_epoch' in request.form:
            epoch = int(request.form['snapshot_epoch'])


        job = ImageClassificationEvaluationJob( 
            name=modelJob._name+"-accuracy-evaluation",
            modeljob_id= modelJob.id(),
            model_epoch= epoch
            )
        

        dataset = job.model_job.train_task().dataset
        if dataset.val_db_task() != None:
            job.tasks.append(
                    tasks.CaffeAccuracyTask(                    
                        job_dir         = job.dir(),
                        job             = job,
                        snapshot        = job.snapshot_filename,
                        db_task         = dataset.val_db_task()
                        )
                    )
        if dataset.test_db_task() != None: 
            job.tasks.append(
                    tasks.CaffeAccuracyTask(                    
                        job_dir         = job.dir(),
                        job             = job,
                        snapshot        = job.snapshot_filename,
                        db_task         = dataset.test_db_task()
                        )
                    )
 

        scheduler.add_job(job)
        return redirect(url_for('evaluations_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

 
def show(job):
    """
    Called from digits.evaluation.views.evaluations_show()
    """
    return render_template('evaluations/images/classification/show.html', job=job)
