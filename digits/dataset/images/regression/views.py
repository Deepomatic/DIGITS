    # Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import random
import shutil
import flask
import json

from digits import utils
from digits.utils.routing import request_wants_json
from digits.webapp import app, scheduler, autodoc
from digits.dataset import tasks
from forms import ImageRegressionDatasetForm
from job import ImageRegressionDatasetJob

NAMESPACE = '/datasets/images/regression'

def find_elements(describ_path):
    """
    Take in a param the file that describ the dataset
    return a dictionnary such as
    data type (datas/labels), index, and format (data or float_data)
    """
    description_fd = open(describ_path, "r")
    description_json = json.loads(description_fd.read())
    if description_json.has_key("path"):
        path = description_json["path"]
    else:
        path = "/".join(describ_path.split('/')[:-1])

    fields = []
    for key in description_json["data"]:
        if "type" in description_json["data"][key]:
            if description_json["data"][key]["type"] == "img":
                fields += ["data_img", "img_target"]

    labels_type = {}
    for label in description_json["labels"]:
        fields += [label]
        labels_type[label] = description_json["labels"][label]["type"]

    ret = {"labels" : {}, "data" : {}}
    i = 0
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            if path.endswith(".json"):
                with open(path, "r") as fd:
                    content = json.loads(fd.read())
                    for key in content.keys():
                        flag = False
                        tmp = []
                        if key == "img_target":
                            imgPath = os.path.join(dirname, content["img_target"])
                            if os.path.exists(imgPath) and os.path.isfile(imgPath):
                                tmp = [("data", i, imgPath)]
                                flag = True
                        elif key in labels_type:
                            if labels_type[key] == "vector":
                                tmp = [("float_data", i, content[key])]
                            flag = True

                        if flag and len(tmp):
                            #REFACTOR
                            if description_json["labels"].has_key(key):
                                if ret["labels"].has_key(key):
                                    ret["labels"][key] += tmp
                                else:
                                    ret["labels"][key] = tmp
                            else:
                                if ret["data"].has_key(key):
                                    ret["data"][key] += tmp
                                else:
                                    ret["data"][key] = tmp
                i += 1
    description_fd.close()
    return ret

def generate_file_output(input, isPreprocessFile = False):
    if not isPreprocessFile:
        try:
            if type(input[0][2]) == list:
                return "\n".join(["{} {} {}".format(kind, idx, " ".join([str(d) for d in value])) for kind, idx, value in input])
            return "\n".join(["{} {} {}".format(kind, idx, value) for kind, idx, value in input])
        except IndexError: #Usually TEST
            return ""
    else:
        return "\n".join(["{}".format(value) for kind, idx, value in input])

def generate_advanced_lmdb_data(job, form, elements):
    job.labels_file = form.textfile_folderPath.data + "/dataset.json"

    number_images = len(elements["data"].itervalues().next())
    percent_val = int(form.percent_val.data)
    percent_test = int(form.percent_test.data)
    percent_train = 100 - percent_val - percent_test
    tmp_path = "/tmp/" + job.dir().split("/")[-1]
    os.mkdir(tmp_path)

    generated_files = {"prepare" : [], "data": {"val":[], "train":[], "test":[]}, "labels": {"val":[], "train":[], "test":[]}}
    for i, key in enumerate(elements["data"]):
        content = elements["data"][key]
        old_content = content
        if content[0][0] == "data": #do it also for labels?
            with open(os.path.join(tmp_path, "files_{}.txt".format(i)), "w") as fd: #base input
                fd.write(generate_file_output(content, True)) #LIST OF TMP FILE
                generated_files["prepare"].append(os.path.join(tmp_path, "files_{}.txt".format(i)))

            new_content = []
            for line in content:
                tmp = line[2].split(" ")
                tmp[0] = tmp_path + "/" + tmp[0]
                new_content.append((line[0], line[1], " ".join(tmp)))
            content = new_content

            with open(os.path.join(tmp_path, "files_{}.txt.tmp".format(i)), "w") as fd: #base input
                fd.write(generate_file_output(content, True)) #LIST OF TMP FILE

        val = old_content[:int(number_images * percent_val * 0.01)]
        test = old_content[len(val):len(val) + int(number_images * percent_test * 0.01)]
        train = old_content[len(val) + len(test):]
        with open(os.path.join(job.dir(), utils.constants.TRAIN_FILE + str(i)), "w") as fd:
            fd.write(generate_file_output(train))
        with open(os.path.join(job.dir(), utils.constants.VAL_FILE + str(i)), "w") as fd:
            fd.write(generate_file_output(val))
        with open(os.path.join(job.dir(), utils.constants.TEST_FILE + str(i)), "w") as fd:
            fd.write(generate_file_output(test))

        val = content[:int(number_images * percent_val * 0.01)]
        test = content[len(val):len(val) + int(number_images * percent_test * 0.01)]
        train = content[len(val) + len(test):]
        #with tmp path
        with open(os.path.join(job.dir(), utils.constants.TRAIN_FILE + str(i) + ".tmp"), "w") as fd:
            fd.write(generate_file_output(train))
            generated_files["data"]["train"].append(os.path.join(job.dir(), utils.constants.TRAIN_FILE + str(i) + ".tmp"))
        with open(os.path.join(job.dir(), utils.constants.VAL_FILE + str(i) + ".tmp"), "w") as fd:
            fd.write(generate_file_output(val))
            generated_files["data"]["val"].append(os.path.join(job.dir(), utils.constants.VAL_FILE + str(i) + ".tmp"))
        if len(test):
            with open(os.path.join(job.dir(), utils.constants.TEST_FILE + str(i) + ".tmp"), "w") as fd:
                fd.write(generate_file_output(test))
                generated_files["data"]["test"].append(os.path.join(job.dir(), utils.constants.TEST_FILE + str(i) + ".tmp"))

        with open(os.path.join(tmp_path, "files_tmp_{}.txt".format(i)), "w") as fd:
            fd.write(generate_file_output(content, True)) #LIST OF TMP FILE

    for i, key in enumerate(elements["labels"]):
        content = elements["labels"][key]
        val = content[:int(number_images * percent_val * 0.01)]
        test = content[len(val):len(val) + int(number_images * percent_test * 0.01)]
        train = content[len(val) + len(test):]

        with open(os.path.join(job.dir(), utils.constants.TRAIN_FILE + str(i) + "_label"), "w") as fd:
            fd.write(generate_file_output(train))
            generated_files["labels"]["train"].append(os.path.join(job.dir(), utils.constants.TRAIN_FILE + str(i) + "_label"))
        with open(os.path.join(job.dir(), utils.constants.VAL_FILE + str(i) + "_label"), "w") as fd:
            fd.write(generate_file_output(val))
            generated_files["labels"]["val"].append(os.path.join(job.dir(), utils.constants.VAL_FILE + str(i) + "_label"))

        if len(test):
            with open(os.path.join(job.dir(), utils.constants.TEST_FILE + str(i) + "_label"), "w") as fd:
                fd.write(generate_file_output(test))
                generated_files["labels"]["test"].append(os.path.join(job.dir(), utils.constants.TEST_FILE + str(i) + "_label"))

    image_folder = None
    for files in generated_files["prepare"]:
        prepare_task = tasks.PrepareFiles(
                job_dir     = job.dir(),
                input_file  = files,
                output_file = files + ".tmp",
                resize_mode = job.resize_mode,
                mean_file   = os.path.join(job.dir(), utils.constants.MEAN_FILE_CAFFE),
                image_dims  = job.image_dims,
                encoding    = form.encoding.data
                )
        job.tasks.append(
                prepare_task
            )


    create_db_task = []
    for type in ("data", "labels"):
        for task_type in ("train", "val", "test"):
            if len(generated_files[type][task_type]):
                for i, file in enumerate(generated_files[type][task_type]):
                    create_db_task.append(
                    tasks.CreateDbTaskRegression(
                        job_dir     = job.dir(),
                        input_file  = file,
                        db_name     = "{}_{}_{}".format(type, task_type, i),
                        image_dims  = job.image_dims,
                        image_folder= image_folder,
                        resize_mode = job.resize_mode,
                        encoding    = form.encoding.data,
                        mean_file   = os.path.join(job.dir(), utils.constants.MEAN_FILE_CAFFE),
                        labels_file = form.textfile_folderPath.data + "/dataset.json",
                        shuffle     = False, #HACK shuffle,
                        parents      = prepare_task
                        )
                    )
                    job.tasks.append(create_db_task[-1])

    # job.tasks.append(
    #     tasks.ClearFiles(
    #         job_dir = job.dir(),
    #         tmp_folder = tmp_path,
    #         parents = create_db_task
    #         )
    #     )


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
    old_content = content# In order to keep the real path
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

    with open(os.path.join(job.dir(), utils.constants.TRAIN_FILE + ".tmp"), "w") as fd:
        fd.write("\n".join([label] + data_train))

    with open(os.path.join(job.dir(), utils.constants.VAL_FILE + ".tmp"), "w") as fd:
        fd.write("\n".join([label] + data_val))

    with open(os.path.join(job.dir(), utils.constants.TEST_FILE + ".tmp"), "w") as fd:
        fd.write("\n".join([label] + data_test))


    data_val = old_content[:int(number_images * percent_val * 0.01)]
    data_test = old_content[len(data_val):len(data_val) + int(number_images * percent_test * 0.01)]
    data_train = old_content[len(data_val) + len(data_test):]
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
                input_file  = utils.constants.TRAIN_FILE + ".tmp",
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
                input_file  = utils.constants.VAL_FILE + ".tmp",
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
                input_file  = utils.constants.TEST_FILE + ".tmp",
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
    shutil.copyfile(form.textfile_labelsPath.data, os.path.join(job.dir(), utils.constants.LABELS_FILE))
    generic(job, form, form.textfile_filesPath.data, os.path.join(job.dir(), utils.constants.LABELS_FILE))

def advanced_format(job, form):
    datasetPath = form.textfile_folderPath.data + "/dataset.json"
    elements = find_elements(datasetPath)
    generate_advanced_lmdb_data(job, form, elements)

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
        elif form.method.data == 'advanced':
            advanced_format(job, form)

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
