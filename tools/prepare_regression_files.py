import numpy as np
import os
import sys
import logging
import PIL.Image
# 82.Image as PIL
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
import traceback

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

from caffe.proto import caffe_pb2

import thread
from threading import Lock
import Queue
import time

flag = False
mean = None
mean_count = 0
lock = Lock()
logger = logging.getLogger('digits.tools.prepare_regression_files')

def process_image(input, output, image_height, image_width, resize_mode, encoding):
    global mean, mean_count, flag, lock
    img = PIL.Image.open(input)
    img.resize((int(image_height), int(image_width)))
    tmp = "/".join(output.split('/')[:-1])
    try:
        lock.acquire()
        mean += img
        mean_count += 1
        if not os.path.exists(tmp):
            os.makedirs(tmp)
    finally:
        lock.release()
    img.save(output, encoding)

def preprocess_files(output_file, input_file, resize_mode, mean_file, image_width, image_height, encoding = "jpeg"):
    global mean, mean_count, flag
    mean = np.zeros((int(image_height), int(image_width), 3), np.float64)
    output_list = []

    with open(output_file, "r") as fd:
        for i, line in enumerate(fd):
            output_list.append(line.split(" ")[0].replace('\n', ''))

    pool = ThreadPool(15)
    rq = []
    with open(input_file, "r") as fd:
        for i, line in enumerate(fd):
            if i == 0:
                continue
            line = line.split(" ")[0]
            if not os.access(line.replace('\n', ''), os.R_OK):
               logger.error("Can't open file:{}".format(line))
               sys.exit(-1)

            params = [line.replace('\n', ''), output_list[i], image_height, image_width, resize_mode, encoding]
            rq.append(pool.apply_async(process_image, params))

    pool.close()
    pool.join()

    logger.debug("Processing images")

    if mean is not None:
        mean = np.around(mean / mean_count).astype(np.uint8)
        data = mean
        # Transform to caffe's format requirements
        if data.ndim == 3:
            # Transpose to (channels, height, width)
            data = data.transpose((2,0,1))
            if data.shape[0] == 3:
                # channel swap
                # XXX see issue #59
                data = data[[2,1,0],...]
        elif mean.ndim == 2:
            # Add a channels axis
            data = data[np.newaxis,:,:]

        blob = caffe_pb2.BlobProto()
        blob.num = 1
        blob.channels, blob.height, blob.width = data.shape
        blob.data.extend(data.astype(float).flat)

        with open(mean_file, 'w') as outfile:
            outfile.write(blob.SerializeToString())
            logger.info('Mean saved at "%s"' % mean_file)

if __name__ == "__main__":
    if len(sys.argv) < 8:
        sys.exit(0)
    output_file = sys.argv[1]
    input_file = sys.argv[2]
    resize_mode = sys.argv[3]
    mean_file = sys.argv[4]
    image_width = sys.argv[5]
    image_height = sys.argv[6]
    encoding = sys.argv[7]
    preprocess_files(output_file, input_file, resize_mode, mean_file, image_width, image_height, encoding)
