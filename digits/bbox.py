import numpy as np
caffe_root = '/home/david/caffe/dsitribute/'
import sys
sys.path.insert(0, caffe_root + 'python')
# cdll.LoadLibrary(caffe_root + 'lib/libcaffe-nv.so.0')
import caffe
from caffe.detector import Detector
import img_utils
import copy
import os
import heapq
