# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.
# -*- coding: utf-8 -*-

import flask
import werkzeug.exceptions

from digits.evaluation import tasks
from digits.webapp import app, scheduler
from job import ImageClassificationEvaluationJob


NAMESPACE = '/evaluations/images/classification'


@app.route(NAMESPACE + '/new', methods=['GET'])
def image_classification_evaluation_new():
    return flask.render_template('evaluations/images/classification/new.html')


@app.route(NAMESPACE + '/accuracy', methods=['POST'])
def image_classification_evaluation_create():
    """Creates a classification performance evaluation task """

    modelJob = scheduler.get_job(flask.request.args['job_id'])

    if modelJob is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    job = None
    try:
        # We retrieve the selected snapshot from the epoch and the train task
        epoch = -1
        if 'snapshot_epoch' in flask.request.form:
            epoch = float(flask.request.form['snapshot_epoch'])

        job = ImageClassificationEvaluationJob(
            name=modelJob._name + "-accuracy-evaluation-epoch-" + str(epoch),
            model_id= modelJob.id(),
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
        return flask.redirect(flask.url_for('evaluations_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise


def show(job, related_jobs=None):
    """
    Called from digits.evaluation.views.evaluations_show()
    """
    return flask.render_template('evaluations/images/classification/show.html', job=job, related_jobs=related_jobs)

