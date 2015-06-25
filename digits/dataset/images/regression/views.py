# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import random

import flask

from digits import utils
from digits.utils.routing import request_wants_json
from digits.webapp import app, scheduler, autodoc
from digits.dataset import tasks
from forms import ImageRegressionDatasetForm
from job import ImageRegressionDatasetJob

NAMESPACE = '/datasets/images/regression'


def generic(job, form, files, labels):
    """
    Add tasks for creating a dataset by reading textfiles
    """
    ### labels


    job.labels_file = labels

    shuffle = bool(form.textfile_shuffle.data)
    fd = open(files, "r")
    content = fd.read().split('\n')
    label = content[0]
    content = content[1:]
    if shuffle:
        random.shuffle(content)
    number_images = len(content)
    fd.close()

    job.number_labels = label.split(' ')

    tmp_path = "/tmp/" + job.dir().split("/")[-1]
    new_content = []
    for line in content:
        tmp = line.split(" ")
        tmp[0] = tmp_path + "/" + tmp[0]
        new_content.append(" ".join(tmp))

    content = new_content
    percent_val = int(form.percent_val.data)
    percent_test = int(form.percent_test.data)
    percent_train = 100 - percent_val - percent_test

    data_val = content[:int(number_images * percent_val * 0.01)]
    data_test = content[len(data_val):len(data_val) + int(number_images * percent_test * 0.01)]
    data_train = content[len(data_val) + len(data_test):]

    os.mkdir(tmp_path)
    with open(os.path.join(tmp_path, "files.txt"), "w") as fd:
        fd.write("\n".join(content))

    with open(os.path.join(job.dir(), utils.constants.TRAIN_FILE), "w") as fd:
        fd.write("\n".join([label] + data_train))

    with open(os.path.join(job.dir(), utils.constants.VAL_FILE), "w") as fd:
        fd.write("\n".join([label] + data_val))

    with open(os.path.join(job.dir(), utils.constants.TEST_FILE), "w") as fd:
        fd.write("\n".join([label] + data_test))

    image_folder = None

    prepare_task = tasks.PrepareFiles(
            job_dir     = job.dir(),
            input_file  = files,
            output_file = os.path.join(tmp_path, "files.txt"),
            resize_mode = job.resize_mode,
            mean_file   = os.path.join(job.dir(), utils.constants.MEAN_FILE_CAFFE),
            image_dims  = job.image_dims,
            encoding    = form.encoding.data
            )
    job.tasks.append(
            prepare_task
        )

    createDbList = [
        tasks.CreateDbTaskRegression(
                job_dir     = job.dir(),
                input_file  = utils.constants.TRAIN_FILE,
                db_name     = utils.constants.TRAIN_DB,
                image_dims  = job.image_dims,
                image_folder= image_folder,
                resize_mode = job.resize_mode,
                encoding    = form.encoding.data,
                mean_file   = os.path.join(job.dir(), utils.constants.MEAN_FILE_CAFFE),
                labels_file = os.path.join(job.dir(), utils.constants.LABELS_FILE),
                shuffle     = shuffle,
                parents      = prepare_task
                ),
        tasks.CreateDbTaskRegression(
                job_dir     = job.dir(),
                input_file  = utils.constants.VAL_FILE,
                db_name     = utils.constants.VAL_DB,
                image_dims  = job.image_dims,
                image_folder= image_folder,
                resize_mode = job.resize_mode,
                encoding    = form.encoding.data,
                mean_file   = os.path.join(job.dir(), utils.constants.MEAN_FILE_CAFFE),
                labels_file = os.path.join(job.dir(), utils.constants.LABELS_FILE),
                shuffle     = shuffle,
                parents      = prepare_task
                ),
    ]

    if len(data_test) > 0:
        createDbList.append(
            tasks.CreateDbTaskRegression(
                job_dir     = job.dir(),
                input_file  = utils.constants.TEST_FILE,
                db_name     = utils.constants.TEST_DB,
                image_dims  = job.image_dims,
                image_folder= image_folder,
                resize_mode = job.resize_mode,
                encoding    = form.encoding.data,
                mean_file   = os.path.join(job.dir(), utils.constants.MEAN_FILE_CAFFE),
                labels_file = os.path.join(job.dir(), utils.constants.LABELS_FILE),
                shuffle     = shuffle,
                parents      = prepare_task
                )
            )

    for t in createDbList:
        job.tasks.append(t)

    job.tasks.append(
        tasks.ClearFiles(
            job_dir = job.dir(),
            tmp_folder = tmp_path,
            parents = [prepare_task] + createDbList
            )
        )

def from_files(job, form):
    files = os.path.join(job.dir(), utils.constants.TMP_FILE)
    labels = os.path.join(job.dir(), utils.constants.LABELS_FILE)
    flask.request.files[form.input_files.name].save(
            files
            )

    flask.request.files[form.input_labels.name].save(
        labels
        )
    
    generic(job, form, files, labels)

def from_path(job, form):
    generic(job, form, form.textfile_filesPath.data, form.textfile_labelsPath.data)

@app.route(NAMESPACE + '/new', methods=['GET'])
@autodoc('datasets')
def image_regression_dataset_new():
    """
    Returns a form for a new ImageRegressionJob
    """
    form = ImageRegressionDatasetForm()
    return flask.render_template('datasets/images/regression/new.html', form=form)

@app.route(NAMESPACE + '.json', methods=['POST'])
@app.route(NAMESPACE, methods=['POST'])
@autodoc(['datasets', 'api'])
def image_regression_dataset_create():
    """
    Creates a new ImageClassificationDatasetJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = ImageRegressionDatasetForm()
    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('datasets/images/regression/new.html', form=form), 400
    job = None
    try:
        job = ImageRegressionDatasetJob(
                name        = form.dataset_name.data,
                image_dims  = (
                    int(form.resize_height.data),
                    int(form.resize_width.data),
                    int(form.resize_channels.data),
                    ),
                resize_mode = form.resize_mode.data
                )

        if form.method.data == 'regression':
            from_files(job, form)
        elif form.method.data == 'upload':
            from_path(job, form)

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        return flask.redirect(flask.url_for('datasets_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job):
    """
    Called from digits.dataset.views.datasets_show()
    """
    return flask.render_template('datasets/images/regression/show.html', job=job)

