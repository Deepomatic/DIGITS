import numpy as np
import matplotlib.pyplot as plt
caffe_root = '/home/DIGITS/caffe/distribute/'
import sys
from ctypes import *
sys.path.insert(0, caffe_root + 'python')
# cdll.LoadLibrary(caffe_root + 'lib/libcaffe-nv.so.0')

import caffe
from caffe.detector import Detector
import urllib
import img_utils
import copy
import PIL.Image as Image

caffe.set_device(3)
caffe.set_mode_gpu()

def jaccard_mod(a,b):
    r1_x1 = a[0]
    r1_y1 = a[1]
    r1_x2 = a[2]
    r1_y2 = a[3]
    r1_width = r1_x2 - r1_x1
    r1_height = r1_y2 - r1_y1
    r2_x1 = b[0]
    r2_y1 = b[1]
    r2_x2 = b[2]
    r2_y2 = b[3]
    r2_width = r2_x2 - r2_x1
    r2_height = r2_y2 - r2_y1
    if r1_width < 0 or r1_height < 0 or r2_width < 0 or r2_height < 0:
        return 0.0
    else :
        if r1_x1 > r2_x2 or r2_x1 > r1_x2 or r1_y1 > r2_y2 or r2_y1 > r1_y2:
            return 0.0
        else :
            max_gauche = max(r1_x1, r2_x1)
            min_droit = min(r1_x1 + r1_width, r2_x1 + r2_width)
            max_bas = max(r1_y1, r2_y1)
            min_haut = min(r1_y1 + r1_height, r2_y1 + r2_height)
            intersection = (min_droit - max_gauche) * (min_haut - max_bas)
            if intersection == (r1_width * r1_height + r2_width * r2_height):
                return 1
            else :
                return ((intersection / (r1_width * r1_height + r2_width * r2_height - intersection)**2))

def jaccard(a,b):
    r1_x1 = a[0]
    r1_y1 = a[1]
    r1_x2 = a[2]
    r1_y2 = a[3]
    r1_width = r1_x2 - r1_x1
    r1_height = r1_y2 - r1_y1
    r2_x1 = b[0]
    r2_y1 = b[1]
    r2_x2 = b[2]
    r2_y2 = b[3]
    r2_width = r2_x2 - r2_x1
    r2_height = r2_y2 - r2_y1
    if r1_width < 0 or r1_height < 0 or r2_width < 0 or r2_height < 0:
        return 0.0
    else :
        if r1_x1 > r2_x2 or r2_x1 > r1_x2 or r1_y1 > r2_y2 or r2_y1 > r1_y2:
            return 0.0
        else :
            max_gauche = max(r1_x1, r2_x1)
            min_droit = min(r1_x1 + r1_width, r2_x1 + r2_width)
            max_bas = max(r1_y1, r2_y1)
            min_haut = min(r1_y1 + r1_height, r2_y1 + r2_height)
            intersection = (min_droit - max_gauche) * (min_haut - max_bas)
            if intersection == (r1_width * r1_height + r2_width * r2_height):
                return 1.0
            else :
                return ((intersection / (r1_width * r1_height + r2_width * r2_height - intersection)))

def merge_boxes(bb_all,proba_all,seuil_percentile,t,function_dist):
    bb_all = bb_all[proba_all>=np.percentile(proba_all,seuil_percentile)]
    proba_all = proba_all[proba_all>=np.percentile(proba_all,seuil_percentile)]
    while len(bb_all)>1 :
        matrix_all = np.zeros((len(bb_all),len(bb_all)))+t
        argmin_matrix = np.zeros(len(bb_all))+t
        min_matrix = np.zeros(len(bb_all))+t
        for i in range(len(bb_all)):
            for j in range(i+1,len(bb_all)):
                matrix_all[i,j] = function_dist(bb_all[i],bb_all[j])
            argmin_matrix[i] = np.argmax(matrix_all[i])
            min_matrix[i] = np.max(matrix_all[i])
        min_bb = np.max(min_matrix)
        i_min = int(np.argmax(min_matrix))
        j_min = int(argmin_matrix[np.argmax(min_matrix)])
        if min_bb > t:
            bb_all = np.vstack((bb_all[list(set(range(len(bb_all)))-set([i_min,j_min]))],(bb_all[i_min] + bb_all[j_min])/2))
            proba_all = np.array(list(proba_all[list(set(range(len(proba_all)))-set([i_min,j_min]))])+[(proba_all[i_min] + proba_all[j_min])/2])
        else :
            break
    return bb_all,proba_all

def add_bb(bb_final,bb_sans,proba_all_sans,bb_temp,proba_temp,seuil_recouvrement,conf,bool_multiboxes=1):
    if len(bb_temp)<2 or not bool_multiboxes:
        if conf:
            return bb_final, conf
        else :
            return bb_final
    else :
        count = 0
        while len(bb_temp)>1:
            bb_sans = np.zeros((1,4))
            proba_all_sans = []
            count_sans = 0
            for i in range(len(bb_temp)):
                jacc = jaccard(bb_temp[i],bb_temp[np.argmax(proba_temp)])
                if jacc<seuil_recouvrement:
                    count_sans += 1
                    if count_sans==1:
                        bb_sans[0] = bb_temp[i]
                    else:
                        bb_sans = np.vstack((bb_sans,bb_temp[i]))
                    proba_all_sans.append(proba_temp[i])
                else :
                    pass
            if (len(bb_sans) > 0 and count_sans>0):
                bb_final = np.vstack((bb_final,bb_sans[np.argmax(proba_all_sans)]))
                if conf:
                    conf.append(np.max(proba_all_sans))
                proba_temp = proba_all_sans
                bb_temp = bb_sans.copy()
            else :
                break
            count += 1
        if conf:
            return bb_final, conf
        else :
            return bb_final
def box_digits(d_c,m_c,net_regression,im_base,mean_regression,nb_classe=1,nb_scales=2,debut_scale=0,scale_ratio=1.5,seuil_recouvrement=0.25,ind_f=0,t=0.00003,seuil_percentile=90,seuil_abs=0.98,bool_multiboxes=1):
    
    if ind_f:
        dist_function = jaccard
    else :
        dist_function = jaccard_mod

    const_mean_imagenet = "/datasets/OverFeat/Digits/classification/imagenet_mean.npy"

    detector = Detector(d_c, m_c, 
                        mean = np.load(const_mean_imagenet).mean(1).mean(1), 
                        raw_scale = 255, 
                        channel_swap = (2,1,0))


    # net_regression = caffe.Net(d_r, m_r, caffe.TEST)
    
    if im_base.isGrayscaleFromMode():
        im_base.convert('RGB')
    im_resize = copy.deepcopy(im_base)
    img = im_resize.toNumpyArray()
    im_show = Image.fromarray(np.uint8(img))
    h_show = img.shape[0]
    w_show = img.shape[1]
    vector = np.array([w_show,h_show,w_show,h_show])
    debut_size = 275

    size = debut_size

    nb_scales = 2
    debut_scale = 0
    scale_ratio = 1.5

    proba_dict = {}
    bb_dict = {}

    for s in range(debut_scale,debut_scale+nb_scales):
        size = int(size*(scale_ratio**s))
        out = detector.detect_scanning(im_base,(size,size),0.98)
        proba_dict['scale-'+str(s)] = out[0][0]
        bb_temp = out[1]
        print(s)
        for key in bb_temp.keys():
            if not bb_dict.has_key(key):
                bb_dict[key] = bb_temp[key]
            else :
                bb_dict[key]+=bb_temp[key]
    print(len(bb_dict[key]))

    bb_final_all = np.zeros((1,4))
    basis = 227
    pad_size = 16
    vect_size = np.mod(range(1,5),2)*basis + np.mod(range(4),2)*basis
    proba_all = []
    nb_classe = 1
    shape = (basis,basis)
    bb_all_dict = {}
    for num_classe in range(nb_classe):
        if bb_dict.has_key('class-'+str(num_classe)):
            bb_class = np.array([b['box'] for b in bb_dict['class-'+str(num_classe)]])*vector
            proba_all = [b['prob'] for b in bb_dict['class-'+str(num_classe)]]
            print(proba_all)
            N = len(proba_all)
            for n in range(N):
                im_scale = copy.deepcopy(im_resize)
                bb = bb_class[n]
                im_scale.crop((int(round(bb[0])),int(round(bb[1])),int(round(bb[2])),int(round(bb[3]))))
                im = np.array(im_scale.toNumpyArray(),dtype=float)/255
                transformer = caffe.io.Transformer({'data': (1,im.shape[2],basis,basis)})
                transformer.set_mean('data', mean_regression.mean(1).mean(1))
                transformer.set_transpose('data', (2,0,1))
                transformer.set_channel_swap('data', (2,1,0))
                transformer.set_raw_scale('data', 255)
                input_image = np.asarray([transformer.preprocess('data', im)])
                bb_n = net_regression.forward_all(data=input_image)['fc9-'+str(num_classe+1)]
                final_bb = bb_n[0] * np.array([im.shape[1],im.shape[0],im.shape[1],im.shape[0]])
                final_bb[0] = max(final_bb[0]+bb[0],0)
                final_bb[1] = max(final_bb[1]+bb[1],0)
                final_bb[2] = min(final_bb[2]+bb[0],w_show)
                final_bb[3] = min(final_bb[3]+bb[1],h_show)
                if not bb_all_dict.has_key('class-'+str(num_classe)):
                    bb_all_dict['class-'+str(num_classe)] = final_bb
                else :
                    bb_all_dict['class-'+str(num_classe)] = np.vstack((bb_all_dict['class-'+str(num_classe)],final_bb))
            if len(bb_all_dict['class-'+str(num_classe)].shape)==1:
                np.expand_dims(bb_all_dict['class-'+str(num_classe)],0)
                print("1 seule box ?")
            bb_all_back = bb_all_dict['class-'+str(num_classe)].copy()
            proba_all_back = proba_all
            print(len(proba_all_back))
            # MERGING OVERFEAT
            bb_all = bb_all_back.copy()
            proba_all = np.array(proba_all_back)
            print(np.mean(proba_all))
            if np.mean(proba_all)>0.99:
                bb_all,proba_all = merge_boxes(bb_all,proba_all,seuil_percentile,t,dist_function)
                bb_final = bb_all[np.argmax(proba_all)]
                conf = [np.max(proba_all)]
                proba_temp = proba_all
                bb_final = np.expand_dims(bb_final,0)
                bb_temp = bb_all.copy()
                bb_sans = []
                proba_all_sans = []
                bb_final,conf = add_bb(bb_final,bb_sans,proba_all_sans,bb_temp,proba_temp,seuil_recouvrement,conf,bool_multiboxes)
            else :
                bb_final = np.zeros(4)
                bb_sans = []
                bb_final = np.expand_dims(bb_final,0)
                print("No sneakers !")
        else :
            bb_final = np.zeros(4)
            bb_sans = []
            bb_final = np.expand_dims(bb_final,0)
            print("No sneakers !")
        bb_final_all = np.vstack((bb_final_all,bb_final/vector))
    bb_final_all = bb_final_all[1:]
    print(bb_final_all)
    print(vector)
    return bb_final_all


if __name__ == "__main__":

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open( "/home/DIGITS/digits/jobs/20150618-191154-b552/mean.binaryproto" , 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    mean_regression = arr[0]

    d_r = "/home/DIGITS/digits/jobs/20150618-194022-d88d/deploy.prototxt"
    m_r = "/home/DIGITS/digits/jobs/20150618-194022-d88d/snapshot_iter_12792.caffemodel"
    im_base = img_utils.Image("/datasets/OverFeat/shoes/tmp.jpg")

    print box_digits("/datasets/OverFeat/Digits/classification/memory_shoes.prototxt","/datasets/OverFeat/Digits/classification/shoes.caffemodel",d_r,m_r,im_base,mean_regression,1)