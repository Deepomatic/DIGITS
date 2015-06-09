# -*- coding: utf-8 -*-

from digits.webapp import app, scheduler, autodoc
from flask import render_template, request, url_for, flash, make_response, abort, jsonify

import digits
import images as evaluation_images
import images.views

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

