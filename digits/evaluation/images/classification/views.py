# -*- coding: utf-8 -*-

from caffe.proto import caffe_pb2
from digits import utils
from digits.config import config_option
from digits.evaluation import tasks
from digits.status import Status
from digits.webapp import app, scheduler
from flask import render_template, request, redirect, url_for, flash
from google.protobuf import text_format
from job import ImageClassificationEvaluationJob

import digits
import numpy as np
import os
import random
import re
import shutil
import sys
import tempfile

NAMESPACE = '/evaluations/images/classification'


@app.route(NAMESPACE + '/new', methods=['GET'])
def image_classification_evaluation_new():
    return render_template('evaluations/images/classification/new.html', form=form)


@app.route(NAMESPACE + '/accuracy', methods=['POST'])
def image_classification_evaluation_create():
    """Creates a classification performance evaluation task """

    modelJob = scheduler.get_job(request.args['job_id'])
 
    if not modelJob:
        return 'Unknown model job_id "%s"' % request.args['job_id'], 500

    job = None
    try:
        # We retrieve the selected snapshot from the epoch and the train task
        epoch = None
        if 'snapshot_epoch' in request.form:
            epoch = int(request.form['snapshot_epoch'])

        job = ImageClassificationEvaluationJob( 
            name=modelJob._name + "-accuracy-evaluation-epoch-" + str(epoch),
            modeljob_id= modelJob.id(),
            model_epoch= epoch
            )
        
        dataset = job.model_job.train_task().dataset

        # We create one task for the validation set if existing
        if dataset.val_db_task() != None:
            job.tasks.append(
                    tasks.CaffeAccuracyTask(                    
                        job_dir         = job.dir(),
                        job             = job,
                        snapshot        = job.snapshot_filename,
                        db_task         = dataset.val_db_task()
                        )
                    )

        # We create one task for the testing set if existing
        if dataset.test_db_task() != None: 
            job.tasks.append(
                    tasks.CaffeAccuracyTask(                    
                        job_dir         = job.dir(),
                        job             = job,
                        snapshot        = job.snapshot_filename,
                        db_task         = dataset.test_db_task()
                        )
                    ) 

        # The job is created
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
