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

caffe.set_device(3)
caffe.set_mode_gpu()

def precisionrecall(conf,labels,nfalseneg=0):
    if not len(conf) :
        rec = 0
        prec = 0
        ap = 0

    sortind = np.argsort(-conf)
    so = conf[sortind]
    tp=labels[sortind]==1
    fp=labels[sortind]!=1
    npos=np.sum(labels)+nfalseneg
    
    fp=np.cumsum(fp)
    tp=np.cumsum(tp)
    rec=np.array(tp,dtype=float)/(npos+np.finfo(float).eps)
    prec=np.array(tp,dtype=float)/(fp+tp)

    ap=0
    for t in np.arange(0,1,0.1):
        if len(prec[rec>=t]):
            p=np.max(prec[rec>=t])
        else :
            p=0
        ap=ap+p/11

    return rec,prec,ap,sortind

def nms(boxes,seuil_jaccard,score):
  if not len(boxes):
    return []
  else :
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2-x1) * (y2-y1)
    I = np.argsort(score)
    vals = np.array(score)[I]

    pick = []
    npicked = 0
    while len(I) > 0:
      last = len(I)-1
      i = I[last]
      pick.append(i)
      suppress = []
      suppress.append(last)
      for pos in range(last):
        j = I[pos]
        xx1 = max(x1[i], x1[j])
        yy1 = max(y1[i], y1[j])
        xx2 = min(x2[i], x2[j])
        yy2 = min(y2[i], y2[j])
        w = xx2-xx1
        h = yy2-yy1
        if w > 0 and h > 0:
          inter = w * h
          o = inter / (area[i] + area[j] - inter)
          if o > seuil_jaccard:
            suppress.append(pos)
      I = np.delete(I,suppress,0)
    return pick

def convert_binaryproto(binaryproto):
    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(binaryproto, 'rb' ).read()
    blob.ParseFromString(data)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    mean = arr[0]

    return mean

def jaccard_mod(a,b):
    r1_x1,r1_y1,r1_x2,r1_y2 = a
    r1_width = r1_x2 - r1_x1
    r1_height = r1_y2 - r1_y1
    r2_x1,r2_y1,r2_x2,r2_y2 = b
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
    r1_x1,r1_y1,r1_x2,r1_y2 = a
    r1_width = r1_x2 - r1_x1
    r1_height = r1_y2 - r1_y1
    r2_x1,r2_y1,r2_x2,r2_y2 = b
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


def addDict(dictionary,elt,key,bool_list):
    if not dictionary.has_key(key):
        if bool_list:
            if isinstance(elt,list):
                dictionary[key] = elt
            else :
                dictionary[key] = [elt]
        else :
            dictionary[key] = elt
    else :
        if bool_list:
            if isinstance(elt,list):
                dictionary[key] += elt
            else :
                dictionary[key].append(elt)
        else :
            dictionary[key] = np.vstack((dictionary[key],elt))

def sliding_window(net_classification,debut_scale,nb_scales,scale_ratio,debut_size,im_base,seuil_prob):
    proba_dict = {}
    bb_dict = {}
    size = debut_size
    for s in range(debut_scale,debut_scale+nb_scales):
        size = int(size*(scale_ratio**s))
        image = copy.deepcopy(im_base)
        image.resizeFixedRatio((size,size), minSize = True)
        w,h = image.size()
        image = np.array(image.toNumpyArray(),dtype=float)/255
        out = net_classification.detect_scanning(image,seuil_prob)
        proba_dict['scale-'+str(s)] = out[0][0]
        bb_temp = out[1]
        for key in bb_temp.keys():
            addDict(bb_dict,bb_temp[key],key,bool_list = True)

    return proba_dict,bb_dict

def f_pad(conv):
    a = 1
    b = a-1
    nb_conv = len(conv)
    for j in range(1,nb_conv+1):
        a = (a-1)*conv[nb_conv-j,1]+conv[nb_conv-j,0]-2*conv[nb_conv-j,2]
        b = (b-1)*conv[nb_conv-j,1]+conv[nb_conv-j,0]-2*conv[nb_conv-j,2]
    
    return a-b

def check_shape(bb_final):
    if len(bb_final.shape)==1:
        return np.expand_dims(bb_final,0)
    else :
        return bb_final

def cropAndTransform(net_regression,im_resize,mean_regression,bb):
    im_scale = copy.deepcopy(im_resize)
    w,h = im_scale.size()
    bb_show = bb*np.array([w,h,w,h])
    im_scale.crop((int(round(bb_show[0])),int(round(bb_show[1])),int(round(bb_show[2])),int(round(bb_show[3]))))
    w_r,h_r = im_scale.size()
    im = np.array(im_scale.toNumpyArray(),dtype=float)/255
    transformer = caffe.io.Transformer({'data': (net_regression.blobs['data'].data.shape)})
    transformer.set_mean('data', mean_regression.mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)
    input_image = np.asarray([transformer.preprocess('data', im)])
    w_s,h_s = input_image.shape[2:]
    ratio = float(w_s)/w_r

    return input_image,(ratio*w,ratio*h)

def regression_boxes(net_regression,bb_all_dict,bb_class,im_resize,num_classe,mean_regression,basis_classification):
    bb_predicted_box = {}
    N = len(bb_class)
    for n in range(N):
        bb = bb_class[n]
        input_image,shape = cropAndTransform(net_regression,im_resize,mean_regression,bb)
        predicted_box = net_regression.forward_all(data=input_image)['fc8-'+str(1)][0] * basis_classification
        h = shape[1]
        w = shape[0]
        pad_w = bb[0] * w
        pad_h = bb[1] * h
        predicted_box[0] = max(predicted_box[0]+pad_w,0.0)
        predicted_box[1] = max(predicted_box[1]+pad_h,0.0)
        predicted_box[2] = min(predicted_box[2]+pad_w,w)
        predicted_box[3] = min(predicted_box[3]+pad_h,h)
        predicted_box /= np.array([w,h,w,h])
        addDict(bb_predicted_box,predicted_box,'class-'+str(num_classe),bool_list = False)

    return check_shape(bb_predicted_box['class-'+str(num_classe)])

def merge_boxes(bbs, probas, t, dist_fun):
    n = len(bbs)
    distances = []
    for i, bb in enumerate(bbs):
        for j in range(i + 1, n):
            d = dist_fun(bb, bbs[j])
            heapq.heappush(distances, (d, i, j))

    keep = np.ones(n, dtype = bool)
    while distances:
        d, i, j = heapq.heappop(distances)

        if keep[i] and keep[j]:
            if d >= t:
                bbs[i] = (bbs[i] + bbs[j]) / 2
                probas[i] = (probas[i] + probas[j]) / 2
                keep[j] = False

                for k, bb in enumerate(bbs):
                    if keep[k] and k != i:
                        d = dist_fun(bbs[i], bbs[k])
                        heapq.heappush(distances, (d, i, k))
            else :
                break

    bbs    = bbs[keep, :]
    probas = np.array(probas)[np.array(keep)]

    return bbs, probas

def add_bb(bb_final,bb_sans,proba_all_sans,bb_temp,proba_temp,seuil_recouvrement,proba_final):
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
            if proba_final:
                proba_final.append(np.max(proba_all_sans))
            proba_temp = proba_all_sans
            bb_temp = bb_sans.copy()
        else :
            break

    return bb_final, proba_final


def multiboxes(bb_all,proba_all,seuil_recouvrement,bool_multiboxes):
    bb_final = bb_all[np.argmax(proba_all)]
    proba_final = [np.max(proba_all)]
    if seuil_recouvrement < 1.0 and bool_multiboxes and len(bb_all)>1:
        bb_temp = bb_all.copy()
        proba_temp = proba_all
        bb_sans = np.zeros((1,4))
        proba_all_sans = []
        bb_final,proba_final = add_bb(bb_final,bb_sans,proba_all_sans,bb_temp,proba_temp,seuil_recouvrement,proba_final)
    return check_shape(bb_final),proba_final


def overfeat_forward_all(net_regression,class_vect,bb_dict,seuil_merge,distance_function,bool_multiboxes,seuil_recouvrement,basis_classification,im_resize,mean_regression):
    bb_all_dict = {}
    proba_all_dict = {}
    bb_merging_dict = {}
    proba_merging_dict = {}
    bb_final_dict = {}
    proba_final_dict = {}
    for num_classe in class_vect:
        if bb_dict.has_key('class-'+str(num_classe)):
            bb_class = np.array([b['box'] for b in bb_dict['class-'+str(num_classe)]])
            proba_all_dict['class-'+str(num_classe)] = [b['prob'] for b in bb_dict['class-'+str(num_classe)]]

            # REGRESSION BOXES
            bb_all_dict['class-'+str(num_classe)] = regression_boxes(net_regression,bb_all_dict,bb_class,im_resize,num_classe,mean_regression,basis_classification)

            # MERGE BOXES
            bb_merging_dict['class-'+str(num_classe)],proba_merging_dict['class-'+str(num_classe)] = merge_boxes(bb_all_dict['class-'+str(num_classe)],proba_all_dict['class-'+str(num_classe)],seuil_merge,distance_function)

            # MULTIBOXES
            bb_final_dict['class-'+str(num_classe)],proba_final_dict['class-'+str(num_classe)] = multiboxes(bb_merging_dict['class-'+str(num_classe)],proba_merging_dict['class-'+str(num_classe)],seuil_recouvrement,bool_multiboxes)

        else :
            print("No shoes!")

    return bb_final_dict,proba_final_dict,bb_merging_dict,bb_all_dict


def box_digits(im_base,d_c,m_c,mean_classification_fd,net_regression,mean_regression,class_vect=[1],nb_scales=3,debut_scale=0,scale_ratio=1.5,seuil_recouvrement=0.25,ind_f=1,seuil_merge=0.9,seuil_prob=0.9,bool_multiboxes=0):
    
    if ind_f:
        distance_function = jaccard
    else :
        distance_function = jaccard_mod

    deploy_classification = d_c
    model_classification = m_c


    mean_classification = convert_binaryproto(mean_classification_fd)
    net_classification = Detector(deploy_classification, model_classification,mean = mean_classification.mean(1).mean(1),raw_scale = 255,channel_swap = (2,1,0))    

    # deploy_regression = d_r
    # model_regression = m_r
    # net_regression = caffe.Net(deploy_regression, model_regression, caffe.TEST)

    im_base = img_utils.Image(np.array(im_base.img))
    if im_base.isGrayscale():
        im_base.convert('RGB')
    im_resize = copy.deepcopy(im_base)

    debut_size = 227 # first scale size
    basis_classification = net_classification.blobs['data'].data.shape
    net_shape = np.array([[11,4,0],[3,2,0],[5,1,2],[3,2,0],[3,1,1],[3,1,1],[3,1,1]],dtype=int)
    pad_size = f_pad(net_shape)
    
    # CLASSIFICATION:
    proba_dict,bb_dict = sliding_window(net_classification,debut_scale,nb_scales,scale_ratio,debut_size,im_base,seuil_prob)

    # REGRESSION:
    bb_final_dict,conf_final_dict,bb_merging_dict,bb_all_dict = overfeat_forward_all(net_regression,class_vect,bb_dict,seuil_merge,distance_function,bool_multiboxes,seuil_recouvrement,basis_classification,im_resize,mean_regression)
  
    bb_final_all = np.zeros((1,4))
    for num_classe in class_vect:
        bb_final_all = np.vstack((bb_final_all,bb_final_dict['class-'+str(num_classe)]))
    bb_final_all = bb_final_all[1:]
    print(bb_final_all)

    return bb_final_all

if __name__ == "__main__":
    
    d_c = "/home/david/caffe/models/david_overfeat/pinterest/classification/memory_deploy.prototxt"
    m_c = "/home/groar/DEEPOMATIC-DIGITS/digits/jobs/20150630-094910-7c6e/snapshot_iter_1370.caffemodel"
    mean_classification_fd = "/home/groar/DEEPOMATIC-DIGITS/digits/jobs/20150630-090906-247c/mean.binaryproto"

    d_r = "/datasets/OverFeat/Digits/regression/deploy.prototxt"
    m_r = "/datasets/OverFeat/Digits/regression/shoes.caffemodel"
    mean_regression = convert_binaryproto("/datasets/OverFeat/Digits/regression/mean.binaryproto")

    im_base = img_utils.Image("/datasets/OverFeat/shoes/tmp.jpg")

    print box_digits(im_base,d_c,m_c,mean_classification_fd,caffe.Net(d_r, m_r, caffe.TEST),mean_regression)