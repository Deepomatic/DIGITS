import numpy as np

caffe_root = '/home/david/caffe/distribute/'
import sys
from ctypes import *
sys.path.insert(0, caffe_root + 'python')
# cdll.LoadLibrary(caffe_root + 'lib/libcaffe-nv.so.0')
import caffe
from caffe.detector import Detector
import img_utils
import copy
import fonctions_overfeat
import PIL

caffe.set_device(2)
caffe.set_mode_gpu()

def box_digits(im_base,d_c,m_c,mean_classification_fd,net_regression,mean_regression,class_vect=[1],nb_scales=3,debut_scale=0,scale_ratio=1.25,seuil_recouvrement=0.25,ind_f=1,seuil_merge=0.9,seuil_prob=0.9,bool_multiboxes=0):
    
    if ind_f:
        distance_function = fonctions_overfeat.jaccard
    else :
        distance_function = fonctions_overfeat.jaccard_mod

    deploy_classification = d_c
    model_classification = m_c


    mean_classification = fonctions_overfeat.convert_binaryproto(mean_classification_fd)
    net_classification = fonctions_overfeat.Detector(deploy_classification, model_classification,mean = mean_classification.mean(1).mean(1),raw_scale = 255,channel_swap = (2,1,0))    

    # deploy_regression = d_r
    # model_regression = m_r
    # net_regression = caffe.Net(deploy_regression, model_regression, caffe.TEST)

    im_base = img_utils.Image(np.array(im_base.img))
    if im_base.isGrayscale():
        im_base.convert('RGB')
    im_resize = copy.deepcopy(im_base)

    debut_size = 275 # first scale size
    basis = 227 # regressor size
    net_shape = np.array([[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1]],dtype=int)
    pad_size = fonctions_overfeat.f_pad(net_shape)
    
    # CLASSIFICATION:
    proba_dict,bb_dict = fonctions_overfeat.sliding_window(net_classification,debut_scale,nb_scales,scale_ratio,debut_size,im_base,seuil_prob)

    # REGRESSION:
    bb_final_dict,_,_,_ = fonctions_overfeat.overfeat_forward_all(net_regression,class_vect,bb_dict,seuil_merge,distance_function,bool_multiboxes,seuil_recouvrement,basis,im_resize,mean_regression)

    bb_final_all = np.zeros((1,4))
    for num_classe in class_vect:
        bb_final_all = np.vstack((bb_final_all,bb_final_dict['class-'+str(num_classe)]))
    bb_final_all = bb_final_all[1:]
    print(bb_final_all)

    return bb_final_all
    # return np.zeros((4,1))

if __name__ == "__main__":
    
    d_c = "/home/david/caffe/models/david_overfeat/pinterest/classification/memory_deploy.prototxt"
    m_c = "/home/groar/DEEPOMATIC-DIGITS/digits/jobs/20150630-094910-7c6e/snapshot_iter_1370.caffemodel"
    mean_classification_fd = "/home/groar/DEEPOMATIC-DIGITS/digits/jobs/20150630-090906-247c/mean.binaryproto"

    d_r = "/datasets/OverFeat/Digits/regression/deploy.prototxt"
    m_r = "/datasets/OverFeat/Digits/regression/shoes.caffemodel"
    mean_regression = fonctions_overfeat.convert_binaryproto("/datasets/OverFeat/Digits/regression/mean.binaryproto")

    im_base = img_utils.Image("/datasets/OverFeat/shoes/tmp.jpg")

    print box_digits(im_base,d_c,m_c,mean_classification_fd,d_r,m_r,mean_regression)