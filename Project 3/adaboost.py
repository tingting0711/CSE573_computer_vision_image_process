import sys
import os
import re

import cv2
import numpy as np
import math
import json

import time
import random
from datetime import timedelta

# import os
# os.chdir("/content/drive/cv_project3")



# pred_v_all = np.load('pred_v_all.npy',allow_pickle=True)
# # print(len(pred_v_all))
# # print(len(pred_v_all[0]))
# bestall = np.load('bestall.npy',allow_pickle=True)
# # print(len(bestall))
# # # print(bestall)
# # exit()

def normalize(img):
    ''' normalize the image/patch, let the min of it become 0, and max become 255, in order to reduce the influence of illumination.
    '''
    min_x = 255
    max_x = 0
    for i, row in enumerate(img):
        for j, num in enumerate(row):
            if(img[i][j]<min_x):
                min_x = img[i][j]
            if(img[i][j]>max_x):
                max_x = img[i][j]
    img = [list(row) for row in img]
    for i, row in enumerate(img):
        for j, num in enumerate(row):
            img[i][j] = (img[i][j]-min_x)/max(0.001,max_x-min_x)
    return np.array(img)
def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if str(type(img))=="<class 'NoneType'>":
        print('lose_path:'+str(img_path))
    if not img.dtype == np.uint8:
        pass

    if show:
        show_image(img)

    img = [list(row) for row in img]
    return img
def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()
def write_image(img, img_saving_path):
    """Writes an image to a given path.
    """
    if isinstance(img, list):
        img = np.asarray(img, dtype=np.uint8)
    elif isinstance(img, np.ndarray):
        if not img.dtype == np.uint8:
            assert np.max(img) <= 1, "Maximum pixel value {:.3f} is greater than 1".format(np.max(img))
            img = (255 * img).astype(np.uint8)
    else:
        raise TypeError("img is neither a list nor a ndarray.")
    cv2.imwrite(img_saving_path, img)
def crop(img, xmin, xmax, ymin, ymax):
    """Crops a given image."""
    if len(img) < xmax:
        print('WARNING')
    patch = img[xmin: xmax]
    patch = [row[ymin: ymax] for row in patch]
    return patch
def get_data(txt_path):
    result = []
    img_pattern = r"/big/img_"
    pos_pattern = r"\."
    data_path = os.listdir(txt_path)
    data_path.sort()
    i = 1
    for img_txt in data_path:
        if img_txt.endswith("txt"):
            with open(str(txt_path)+str(img_txt),'r') as f:
                iname = ''
                num = 0
                face = []
                major_axis_radius = 0
                minor_axis_radius = 0
                angle = 0
                center_x = 0
                center_y = 0
                for line in f:
                    data = str(line.strip())
                    if re.search(img_pattern, data):
                        iname = data
                    elif re.search(pos_pattern, data):
                        data = data.split()
                        major_axis_radius = int(float(data[0]))
                        minor_axis_radius = int(float(data[1]))
                        center_x = int(float(data[3]))
                        center_y = int(float(data[4]))
                        x = center_x-minor_axis_radius
                        y = center_y-major_axis_radius
                        sample = {"iname":iname,"bbox":[int(x), int(y), int(minor_axis_radius),int(major_axis_radius)],"num":int(num)}
                        result.append(sample)
                        print("sample:"+str(sample))
                        print("data:"+str(data))
                        i+=1
                    else :
                        num = int(data)
    result_array = np.array(result)
    np.save('face_list.npy',result_array)
    print('all ' +str(i)+' images')
    exit()
    return result
def get_test_data1(txt_path):
    #get all data(image) inside 'data_path'(including sub path, sub sub path and so on)
    data_path = os.listdir(txt_path)
    gray_test_images = []
    # data_path.sort()
    i = 0
    print('txt_path:'+str(txt_path))
    for img_txt in data_path:
        if img_txt.endswith("txt"):
            with open(str(txt_path)+str(img_txt),'r') as f:
                for line in f:
                    data = str(line.strip())
                    img_path = './originalPics/'+str(data)+'.jpg'
                    img = np.array(read_image(img_path))
                    image = {'iname':img_path,'gray_img':img}

                    gray_test_images.append(image)
                    # # print('data: '+str(data))
                    # nameObj = re.search( r'img_\d{1,}', data, re.M|re.I)
                    # name = nameObj.group()
                    # write_image(img,'./test_data/'+data+'.jpg')
                    i+=1
                    if i%100==0:
                        print('finish '+str(i)+' pictures')
                        # break
        # if i==1000:
        #                 print('finish '+str(i)+' pictures')
        #                 break
    np.save('gray_test_images.npy',gray_test_images)
    print(str(i)+' pictures')
    exit()
    # 2845 pictures
    return gray_test_images
def preprocess(image):
    po = 1
    ne = 1
    # data_path = os.listdir('data/originalp/')
    # data_path.sort()
    # images = []
    # for img_jpg in data_path:
    #     if (img_jpg.endswith(".jpg")) and (not img_jpg.endswith("-.jpg")):
    #         print ('img_jpg'+str(img_jpg))
            # po:8263 -> 7444
    for i in range(len(image)):
        img_path = './originalPics/'+str(image[i]['iname']+'.jpg')
        positive_path = 'data/positive/'
        # negative_path = 'data/negative/'
        img = read_image(img_path)
        num = image[i]['num']
        offx = (image[i]['bbox'][3] - image[i]['bbox'][2])
        ymin = image[i]['bbox'][0]
        ymax = ymin+2*image[i]['bbox'][2]
        xmin = image[i]['bbox'][1]+int(1.5*offx)
        xmax = xmin+2*image[i]['bbox'][2]
        if xmin>=0 and xmax<=len(img) and ymin>=0 and ymax<=len(img[1]):   
            crop_img = read_image('data/originalp/'+str(img_jpg))
            ii = "%04d" % po
            # write_image(crop_img,'data/originalp/'+str(ii)+'.jpg')
            # norm1 = normalize(crop_img)
            # write_image(norm1,'data/11/'+str(ii)+'.jpg')
            resize_img = cv2.resize(np.array(crop_img),(24,24))
            # write_image(resize_img,'data/20/'+str(ii)+'.jpg')
            (mean , stddv) = cv2.meanStdDev(resize_img)
            norm = (resize_img-mean)/stddv
            norm = np.array(normalize(norm))
            cv2.imwrite('data/positive/'+str(ii)+'.jpg', norm)
            # write_image(norm,positive_path+str(ii)+'.jpg')
            po+=1
            # ii = "%04d" % po
            # flip = cv2.flip(norm,1)
            # cv2.imwrite('data/positive/'+str(ii)+'.jpg', flip)
            # po+=1
        # if po>=20:
        # create negative images:
        # # t = 1
        # r = 2*image[i]['bbox'][2]
        # # if num == 1 and ne <5000:
        # if num == 1:
        #     for a in range(int(len(img)/r)):
        #         for b in range(int(len(img[1])/r)):
        #             if (a+1)*r < xmin or r*a>xmax or (b+1)*r < ymin or r*b>ymax :
        #                 if (a+1)*r<len(img) and (b+1)*r<len(img[1]):
        #                     # t+=1
        #                     crop_img = crop(img, a*r, (a+1)*r, b*r, (b+1)*r)
        #                     jj = "%04d" % ne
        #                     write_image(crop_img,'data/0/'+str(jj)+'.jpg')
        #                     resize_img = cv2.resize(np.array(crop_img),(24,24))
        #                     write_image(resize_img,negative_path+str(jj)+'.jpg')
        #                     ne+=1
        #                     # if t>=3:
        #                     #     break;
    print("po:"+str(po))
    # print("ne:"+str(ne))
def integral_image(img):
    sum = img.cumsum(axis=-1).cumsum(axis=-2)
    
    return sum
def get_integral(image_path):
    data_path = os.listdir(image_path)
    data_path.sort()
    images = []
    i = 1
    for img_jpg in data_path:
        if i%1000==0:
            print('have:'+str(i)+' pictures')
        if i>4005:
            break
        i+=1
        if img_jpg.endswith(".jpg"):
            img = np.array(read_image(str(image_path)+str(img_jpg)))/255
            images.append(img)
    integrals = integral_image(np.array(images))
    return integrals         
def create_feature(width,height,stride=1,increase=1):
    features = []
    #----------------------------------------
    '''type 1
       +++---        D   E   F  
       +++---      
       +++---
       +++---
       +++---
       +++---        C   B   A  
       feature = (DEBC) - (AFEB) = (B+D-E-C) - (A+E-B-F) = -A+2B-C+D-2E+F = 2B+D+F-A-C-2E
    '''
    feature1 = []
    for x in range(0,width,stride):
        for y in range(0,height,stride):
            for w in range(1,width-x,2*2*increase):
                for h in range(1,height-y,2*1*increase):
                    A=(x+w,y+h)
                    B=(x+(w//2),y+h)
                    C=(x,y+h)
                    D=(x,y)
                    E=(x+(w//2),y)
                    F=(x+w,y)
                    add = [B,B,D,F]
                    sub = [A,C,E,E]
                    feature1.append((tuple(add), tuple(sub)))
    features.extend(feature1)
    print('finish feature1:'+str(len(feature1)))


    #----------------------------------------
    '''type 2
       ------      D    C
       ------   
       ------      E    B
       ++++++
       ++++++      F    A
       ++++++
       feature = (ABEF)-(BCDE) = (A+E-B-F) - (B+D-E-C) = A-2B+C+2E-D-F= A+C+2E-2B-D-F
    '''
    feature2 = []
    for x in range(0,width,stride):
        for y in range(0,height,stride):
            for w in range(1,width-x,2*1*increase):
                for h in range(1,height-y,2*2*increase):
                    A=(x+w,y+h)
                    B=(x+w,y+(h//2))
                    C=(x+w,y)
                    D=(x,y)
                    E=(x,y+(h//2))
                    F=(x,y+h)
                    add = [A,C,E,E]
                    sub = [B,B,D,F]
                    feature2.append((tuple(add), tuple(sub)))
    features.extend(feature2)
    print('finish feature2:'+str(len(feature2)))


    #----------------------------------------
    '''type 3
       --++--        E  F  G  H
       --++--
       --++--
       --++--
       --++--
       --++--        D  C  B  A
       feature = (BGFC)-[(CFED)+(AHGB)] = (B+F-C-G)-[(C+E-D-F)+(A+G-H-B)] = -A+2B-2C+D-E+2F-2G+H
    '''
    feature3 = []
    for x in range(0,width,stride):
        for y in range(0,height,stride):
            for w in range(1,width-x,3*increase):
                for h in range(1,height-y,increase):
                    A=(x+w,y+h)
                    B=(x+2*(w//3),y+h)
                    C=(x+w//3,y+h)
                    D=(x,y+h)
                    E=(x,y)
                    F=(x+w//2,y)
                    G=(x+2*(w//3),y)
                    H=(x+w,y)
                    add = [B,B,D,F,F,H]
                    sub = [A,C,C,E,G,G]
                    feature3.append((tuple(add), tuple(sub)))
    features.extend(feature3)
    print('finish feature3:'+str(len(feature3)))

    # return features


    #----------------------------------------
    '''type 4
       ++++++      E   D
       ++++++
       ------      F   C
       ------      G   B
       ++++++
       ++++++      H   A
       feature = (CFED)+(AHGB)-(BGFC) = (C+E-D-F)+(A+G-H-B)-(B+F-C-G) = A-2B+2C-D+E-2F+2G-H
    '''
    # feature4 = []
    # for x in range(0,width,stride):
    #     for y in range(0,height,stride):
    #         for w in range(1,width-x,increase):
    #             for h in range(1,height-h,3*increase):
    #                 A=(x+w,y+h)
    #                 B=(x+w,y+2*(h//3))
    #                 C=(x+w,y+(h//3))
    #                 D=(x+w,y)
    #                 E=(x,y)
    #                 F=(x,y+(h//3))
    #                 G=(x,y+2*(h//3))
    #                 H=(x,y+h)
    #                 add = [A,C,C,E,G,G]
    #                 sub = [B,B,D,F,F,H]
    #                 feature4.append((tuple(add), tuple(sub)))
    # features.extend(feature4)
    # print('finish feature4:'+str(len(feature4)))




    #----------------------------------------
    '''type 5
       +++---       
       +++---       I  H  G
       +++---
       ---+++       F  E  D
       ---+++
       ---+++       C  B  A
       feature = (EHIF)+(ADEB)-(BEFC)-(DGHE) = (E-H+I-F)+(A-D+E-B)-(B-E+F-C)-(D-G+H-E) = A-2B+C-2D+4E-2F+G-2H+I
    '''
    # feature5 = []
    # for x in range(0,width,stride):
    #     for y in range(0,height,stride):
    #         for w in range(1,width-x,2*increase):
    #             for h in range(1,height-y,2*increase):
    #                 A=(x+w,y+h)
    #                 B=(x+(w//2),y+h)
    #                 C=(x,y+h)
    #                 D=(x+w,y+(h//2))
    #                 E=(x+w//2,y+(h//2))
    #                 F=(x,y+(h//2))
    #                 G=(x+w,y)
    #                 H=(x+(w//2),y)
    #                 I=(x,y)
    #                 add = [A,C,E,E,E,E,G,I]
    #                 sub = [B,B,D,D,F,F,H,H]
    #                 feature5.append((tuple(add), tuple(sub)))
    # features.extend(feature5)
    # print('finish feature5:'+str(len(feature5)))
    # print('----------------------------------------')
    return features
def constract_adaboost(feature,positive,negative,weight,false_positive_rate=0.3,true_positive_rate=0.99):
    # print("begin training adaboost!")
    # print("face images:"+str(len(positive))+", no-face images:"+str(len(negative))+", feature:"+str(len(feature)))
    weekClass = []
    alpha = []
    m = len(negative)
    l = len(positive)   
    images = np.vstack((positive,negative))
    labels = np.hstack((np.ones(l),np.zeros(m)))
    i=0
    start_time_ada = time.time()
    while True:
        # print('---------------adaboost'+str(i)+':')
        i+=1
        weight = np.array(weight/weight.sum())

        start_time = time.time()
        bestFeature, weightErr, adabEst, bestErr= constract_best_feature(feature,images,labels,weight,i)
        end_time = time.time()
        time_dif = end_time - start_time
        # print("compute one good feature: "+str(i)+" , time usage: " + str(timedelta(seconds=int(round(time_dif)))))

        # print('feature:'+str(bestFeature[0])+', threshold:'+str(bestFeature[1])+', polarity:'+str(bestFeature[2])+', num of error:'+str(bestErr.sum())+',weightErr:'+str(weightErr))
        expon = 1-bestErr
        beta = weightErr/(1.0-weightErr)
        alpha_ = np.log((1.0-weightErr)/max(weightErr,1e-16))
        alpha.append(alpha_)
        weight = np.multiply(np.power(beta,expon),weight)
        weekClass.append(bestFeature)
        (false_posi, true_posi) =calculate_ensemble_error(weekClass,alpha,positive,negative)
        print('---------------adaboost'+str(i)+':, fp_rate:'+str(false_posi)+',tp_rate:'+str(true_posi)+', feature index:'+str(bestFeature[3]))
        if false_posi<=0.10 or true_posi>=0.85:
            print('have '+str(i)+' adaboost feature, fp_rate:'+str(false_posi)+',tp_rate:'+str(true_posi))
            weekClass = np.array(weekClass)
            np.save('weekClass_'+str(true_posi)+'.npy',weekClass)
            alpha = np.array(alpha)
            np.save('alpha_'+str(true_posi)+'.npy',alpha)
            # exit()
            break
    end_time_ada = time.time()
    time_dif_ada = end_time_ada - start_time_ada
    print("finish adaboost in time : " + str(timedelta(seconds=int(round(time_dif_ada)))))
    exit()
    return (alpha,weekClass,weight,tmin,false_posi, true_posi)
def calculate_ensemble_error(weekClass,alpha,positive,negative,cascade=False):
    m = len(negative)
    l = len(positive)
    agg_po = np.zeros(len(positive))
    agg_ne = np.zeros(len(negative))
    for i in range(len(weekClass)):
        features = weekClass[i][0]
        threshold = weekClass[i][1]
        polarity = weekClass[i][2]
        pred_po = pred_feature(features,positive)
        pred_ne = pred_feature(features,negative)
        pred_po_c= pred_class(pred_po,threshold,polarity)
        pred_ne_c = pred_class(pred_ne,threshold,polarity)
        agg_po+=alpha[i]*pred_po_c
        agg_ne+=alpha[i]*pred_ne_c
    # sort_pred_po = np.sort(agg_po)
    # tmin = sort_pred_po[int(l/100)-1]
    true_positive = np.zeros(len(positive))
    false_positive = np.zeros(len(negative))
    if cascade==False:
        T = 0.5*np.array(alpha).sum()
        true_positive[agg_po>=T] = 1
        false_positive[agg_ne>=T] = 1
        return (false_positive.sum()/m, true_positive.sum()/l)
    # true_positive[agg_po>=tmin] = 1
    # false_positive[agg_ne>=tmin] = 1
    # return (false_positive.sum()/m, true_positive.sum()/l, tmin)

def calculate_image_feature_m(index, features, images,labels,weights, step):
    # print('****calculate_image_feature_m, images:'+str(len(images)))
    pred_v_all = []
    bestall = []
    weightErr = []
    num = int(len(images)/2)
    flag=1
    start_time_i = time.time()
    for i, feature in enumerate(features):
        if i%1000==0:
            # print('is calculate_image_feature_m : '+str(i)+' /'+str(len(features)))
            end_time_i = time.time()
            time_dif_i = end_time_i - start_time_i
            print("compute "+str(i)+" feature, time:" + str(timedelta(seconds=int(round(time_dif_i)))))
        add = feature[0]
        sub = feature[1]
        pred_f = pred_feature(feature,images)
        # range_max = posi_f_mean = np.sum(pred_f[0:num])/num
        # range_min = nega_f_mean = np.sum(pred_f[num:2*num])/num
        posi_f_mean = np.sum(pred_f[0:num])/num
        nega_f_mean = np.sum(pred_f[num:])/num
        posi_f_min = np.min(pred_f[0:num])
        nega_f_min = np.min(pred_f[num:])
        posi_f_max = np.max(pred_f[0:num])
        nega_f_max = np.max(pred_f[num:])
        if posi_f_mean>=nega_f_mean:
            polarity='>'
            if nega_f_mean>posi_f_min:
                range_min = nega_f_mean
            else:
                range_min = posi_f_min
            if posi_f_mean<nega_f_max:
                range_max = posi_f_mean
            else:
                range_max=nega_f_max
        else:
            polarity='<'
            if posi_f_mean>nega_f_min:
                range_min = posi_f_mean
            else:
                range_min = nega_f_min
            if nega_f_mean<posi_f_max:
                range_max = nega_f_mean
            else:
                range_max=posi_f_max
        pred_v_best = []
        minErr_i = float('inf')
        best_t = 0
        best_p = '>'
        for j in range(0,len(pred_f),step):
            threshold = pred_f[j]
            if threshold>=range_min and threshold<=range_max:
                pred_v = pred_class(pred_f,threshold,polarity)
                error = np.ones(len(images))
                error[pred_v==labels] = 0
                weightErr = (weights*error).sum()
                if weightErr<minErr_i:
                    pred_v_best= pred_v
                    best_t = threshold
                    best_p = polarity
        pred_v_all.append(pred_v_best)
        bestall.append([feature, best_t, best_p,i])
    pred_v_all = np.array([pred_v_all])
    bestall = np.array(bestall)
    np.save('pred_v_all.npy',pred_v_all)
    np.save('bestall.npy',bestall)
    # print('compute_image_feature_matrix:'+str(index)+)


def constract_best_feature(features,images,labels,weights,index,step=1):
    adabEst = []
    weightErr = []
    bestErr = []
    # pred_v_all = []
    # bestall = []
    # print('begin constract_best_feature:')
    minErr = float('inf')
    # start_time_r = time.time()
    # if index==1:
    #     calculate_image_feature_m(index, features, images,labels,weights,step)
    # end_time_r = time.time()
    # time_dif_r = end_time_r - start_time_r
    # print("image_feature_matrix: "+str(index)+ ", time:" + str(timedelta(seconds=int(round(time_dif_r)))))
    # print('index:'+str(index))
    # print('load :'+str(int((index-1)/recal)))
    # pred_v_all = np.load('pred_v_all.npy')
    # bestall = np.load('bestall.npy')
    start_time_b = time.time()
    for i,feature in enumerate(features):
        # print(len(pred_v_all))
        # print(pred_v_all.shape())
        # print(i)
        # exit()
        pred_v = pred_v_all[i]
        error = np.ones(len(images))
        error[pred_v==labels] = 0
        weightErr = (weights*error).sum()
        if weightErr<minErr:
            minErr = weightErr
            # best = {'feature':bestall[i][0],'threshold':bestall[i][1], 'polarity':bestall[i][2], 'index':i}
            bestFeature = np.array([bestall[i][0], bestall[i][1], bestall[i][2], i])
            ##bestErr if it predicted corectly, then 0; else 1.
            bestErr = error
            ##predict 0 indicate no-face; 1 indicate face.
            adabEst= pred_v
    end_time_b = time.time()
    time_dif_b = end_time_b - start_time_b
    # print("find best feature in "+str(index)+": " + str(timedelta(seconds=int(round(time_dif_b)))) + ', feature index: '+str(bestFeature[3]))
    return (bestFeature,minErr,adabEst,bestErr)
def pred_class(preds,threshold,polarity):
    # print('preds:'+str(len(preds)))
    result = np.ones(len(preds))
    if polarity == '<':
        result[preds[:]>=threshold] = 0
    else:
        result[preds[:]<=threshold] = 0
    return result
def pred_feature(feature,images,amplify=1):
    feature  = feature*amplify
    # print('feature:'+str(type(feature)))
    # print('feature:'+str(feature))
    # print('images:'+str(type(images)))
    # print('pred_feature, images:'+str(len(images))+', '+str(len(images[0]))+', '+str(len(images[0][0])))
    # print(len(images))
    # print(len(images[0]))
    # print(len(images[0][0]))
    # exit()
    # print(type(images))
    # print(len(images))
    # print(len(images[0]))
    # print(len(images[0][0]))
    # images = images.tolist()
    add = feature[0]
    sub = feature[1]
    add_all = images[:,[x[0] for x in add],[y[1] for y in add]].sum(axis=-1)
    sub_all = images[:,[x[0] for x in sub],[y[1] for y in sub]].sum(axis=-1)
    result = add_all - sub_all
    return result
def constract_cascade(feature,positive,negative,false_positive_rate,target_false_positive_rate=0.01):
    # #initialize the weights
    m = len(negative)
    l = len(positive)
    p_weight = (np.ones(l)*1.0)/(2*l)
    n_weight = (np.ones(m)*1.0)/(2*m)
    weight = np.hstack((p_weight,n_weight))
    cascade = []
    # print("begin training cascade!")
    # print("face images:"+str(len(positive))+", no-face images:"+str(len(negative))+", feature:"+str(len(feature))+".")
    j=1
    pos_all_r = neg_all_r = 1
    while True:
        print('------------------stage'+str(j)+'------------------')
        (alpha,weekClass,weight,tmin,false_posi, true_posi) = constract_adaboost(feature,positive,negative,weight,false_positive_rate)
        cascade.append((alpha,weekClass,tmin,false_posi, true_posi))
        pos_all_r *= true_posi
        neg_all_r *= false_posi
        # print('pos_all_r:'+str(pos_all_r)+',neg_all_r:'+str(neg_all_r)+'.tmin:'+str(tmin))
        # print('\n\n')
        if neg_all_r<=target_false_positive_rate:
            break
        j+=1
    return cascade
def cascade_predict(cascade,img):
    agg_pre = 0
    for i in range(len(cascade)):
        alpha = cascade[i][0]
        weekClass = cascade[i][1]
        agg_pred = 0
        tmin = cascade[i][2]
        for j in range(len(weekClass)):
            features = weekClass[j][0]
            threshold = weekClass[j][1]
            polarity = weekClass[j][2]
            pred_f = pred_feature(features,imgs,int(img.shape[0]/24))
            pred_c= pred_class(pred_f,threshold,polarity)
            agg_pred +=alpha[j]*pred_c

        if agg_pred < tmin:
            return 0
        else:
            agg_pred = 1
    return pred_c

def intersect_box(box1,box2):
    x1,y1,size1 = box1
    x2,y2,size2 = box2
    s1 = x1+(size1)/2
    t1 = y1+(size1)/2
    s2 = x2+(size2)/2
    t2 = y2+(size2)/2
    t = (size1+size2)/8
    delta1 = delta2 = 0
    if s1>=s2:
        delta1 = s1-s2
    else:
        delta1 = s2-s1
    if t1>=t2:
        delta2 = t1-t2
    else:
        delta2 = t2-t1


    if delta1<t and delta2<t:
        return True
    else :
        return False

def get_test_data(data_path):
    #get all data(image) inside 'data_path'
    data = os.listdir(data_path)
    data.sort()
    images = []
    imgName = []
    # i = 1
    for img_jpg in data:
        if (img_jpg.endswith(".jpg")):
            img = read_image(data_path+'/'+img_jpg)
            images.append(img)
            imgName.append(img_jpg)
        # if i%100==0:
            print(img_jpg)
            # i+=1
    print('images:'+str(len(images)))

    np.save('test_images.npy',np.array(images))
    np.save('test_names.npy',np.array(imgName))
    return np.array(images), np.array(imgName)

def normalizeVariance(img):    
    (mean , stddv) = cv2.meanStdDev(img)
    stddv = max(stddv, 0.00001)
    norm = (img-mean)/stddv
    norm = np.array(normalize(norm))
    return norm

def cascade_predict(cascade,img):
    agg_pre = 0
    for i in range(len(cascade)):
        alpha = cascade[i][0]
        weekClass = cascade[i][1]
        agg_pred = 0
        tmin = cascade[i][2]
        for j in range(len(weekClass)):
            features = weekClass[j][0]
            threshold = weekClass[j][1]
            polarity = weekClass[j][2]
            pred_f = pred_feature(features,imgs,int(img.shape[0]/24))
            pred_c= pred_class(pred_f,threshold,polarity)
            agg_pred +=alpha[j]*pred_c

        if agg_pred < tmin:
            return 0
        else:
            agg_pred = 1
    return pred_c

def getAdaboost(img, weekClass, alpha, bigSize, stepSize=24, stride=3):

    bbox = []
    alphaT = 0.65*alpha.sum()
    ii = 1

    print('image_shape[0]:'+str(img.shape[0]))
    print('image_shape[1]:'+str(img.shape[1]))
    # print('bigSize:'+str(bigSize))
    # print('stepSize:'+str(stepSize))
    # print('stride:'+str(stride))

    # print(range(1,4,2))
    start_time_face = time.time()
    sizeindex = 1
    for size in range(int(bigSize/10),bigSize,stepSize):
        images = []
        remeberXY = []

        start_time_xy0 = time.time()
        for x in range(0,int(0.5*(img.shape[0]-size)),stride):
            for y in range(0,img.shape[1]-size,stride):
                patch = crop(img, x, x+size, y, y+size)
                if size!=24:
                    # print(type(patch))
                    # print(patch)
                    patch = cv2.resize(np.array(patch),(24,24))
                patch = integral_image(normalizeVariance(patch))
                images.append(patch)
                remeberXY.append([x,y])
        end_time_xy0 = time.time()
        time_dif_xy0 = end_time_xy0 - start_time_xy0
        # print("integral time: " + str(time_dif_xy0))

        images = np.array(images)
        agg_pred = np.zeros(len(images))
        
        # if len(images)==1:
        #     print('only one, with size:'+str(size))

        start_time_xy = time.time()
        for j in range(len(weekClass)):
            feature = weekClass[j][0]
            threshold = weekClass[j][1]
            polarity = weekClass[j][2]
            pred_f = pred_feature(feature,images)
            pred_c= pred_class(pred_f,threshold,polarity)
            # print('pred_c:'+str(pred_c))
            # exit()
            agg_pred +=alpha[j]*pred_c
            ii+=1
        end_time_xy = time.time()
        time_dif_xy = end_time_xy - start_time_xy
        # print('classify time: ' + str(time_dif_xy))

        # print('size:'+str(size)+', remeberXY:'+str(len(remeberXY)))
        start_time_xy1 = time.time()
        for i in range(len(remeberXY)):

            # start_time_xy11 = time.time()

            if agg_pred[i] >= alphaT:
                ibox = [remeberXY[i][0],remeberXY[i][1],size]
                flag = 0
                for j in range(len(bbox)):
                    if intersect_box(ibox,bbox[j][0])==1:
                        bbox[j][1]+=1
                        flag = 1
                        # if agg_pred[i] > bbox[j][2]:
                        #     # bbox[j][0] = ibox
                        #     bbox[j][2] = agg_pred[i]
                        break
                if flag ==0:
                    bbox.append([ibox,1,agg_pred[i]])
            # end_time_xy11 = time.time()
            # time_dif_xy11 = end_time_xy11 - start_time_xy11
            # print('one box: '+str(time_dif_xy11))

        end_time_xy1 = time.time()
        time_dif_xy1 = end_time_xy1 - start_time_xy1
        # print('size:'+str(size)+', imgs:'+str(len(images))+', integral: ' + str(time_dif_xy0)+', classify: ' + str(time_dif_xy)+', box:'+str(time_dif_xy1)+'\n')
        # print('ind:'+str(sizeindex)+', size:'+str(size)+', imgs:'+str(len(images))+', integral: ' + str(time_dif_xy0)+', classify: ' + str(time_dif_xy)+', box:'+str(time_dif_xy1)+'\n')


        sizeindex += 1
    # print('ii:'+str(ii))
    # print('sizeindex:'+str(sizeindex))
    end_time_face = time.time()
    time_dif_face = end_time_face - start_time_face
    print("finish one images, time: " + str(timedelta(seconds=int(round(time_dif_face)))) )
    
    return bbox
def drawRectangle(img, bbox):
    color = []
    # min_v = min(bbox)
    for i in range(len(bbox)):
        if bbox[i][1]>=1:
        # if bbox[i][2]>=0.8*max_v:
            ibox = bbox[i][0]
            color = (int(random.random()*255),int(random.random()*255),int(random.random()*255))
            # img = cv2.rectangle(img, (ibox[0],ibox[1]), (ibox[0]+ibox[2],ibox[1]+ibox[2]), color, 3)
            img = cv2.rectangle(img, (ibox[0],ibox[1]), (ibox[0]+ibox[2],ibox[1]+ibox[2]), (0,0,255), 3)
    return img
def main():
    #python detect.py [datadirectory]
    
    ###### preprocess part, to get train image:positive + negative;
    # get_data('./FDDB-folds/')
    result_array = np.load('face_list.npy')
    result = result_array.tolist()
    preprocess(result_array)  # create 24*24 positive/negative gray image

    ###### create 24*24 positive/negative integral image
    # positive = get_integral('./data/positive/')
    # np.save('positive_it.npy',positive)
    # print(len(positive))#4128
    # negative = get_integral('./data/negative/')
    # np.save('negative_it.npy',negative)
    # exit()
    
    traning  = 0
    stride = 1
    increase = 1
    # a = [{'a':1,'b':2,'c':3}]
    # with open("result.json", "w") as fp:
    #     json.dump(a , fp) 
    # with open("result.json", "r") as fp:
    #     b = json.load(fp)
    # print(b)
    # exit()

    if traning==1:
        ####training cascaded adaboost feature whose all_false_positive_rate is less than 0.001
        # os.chdir("drive/cv_project3/")
        positive_it = np.load('positive_it.npy')
        negative_it = np.load('negative_it.npy')
        weekClass = np.load('weekClass_0.90075.npy',allow_pickle=True)
        alpha = np.load('alpha_0.90075.npy',allow_pickle=True)
        # print('positive_it:'+str(len(positive_it)))
        false_positive = 0.3
        feature = create_feature(24,24,stride,increase)
        print('stride:'+str(stride)+', increase:'+str(increase)+', feature:'+str(len(feature)))
        print('----------------------------------------\n')
        cascade = constract_cascade(feature,positive_it[0:4000],negative_it[0:4000],false_positive)
        cascade = np.array(cascade)
        np.save('cascade2255.npy',cascade)
    elif traning==0:
        ####testing 
        # cascade = np.load('cascade.npy')
        # data_path = str(sys.argv[1])
        data_path = './test_images_1000/'
        result = []

        start_time_xy = time.time()
        weekClass = np.load('weekClass_0.90075.npy')
        alpha = np.load('alpha_0.90075.npy')
        end_time_xy = time.time()
        time_dif_xy = end_time_xy - start_time_xy
        print('load time: ' + str(time_dif_xy))


        print('weekClass:'+str(len(weekClass)))
        data = os.listdir(data_path)
        data.sort()
        jj = 0
        for imgName in data:
            if (imgName.endswith(".jpg")):
                jj+=1
                print('imgName:'+str(imgName))
                img = np.array(read_image(data_path+'/'+imgName))
                
                start_time_xy = time.time()
                # img = normalizeVariance(img)
                # img = integral_image(img)
                end_time_xy = time.time()
                time_dif_xy = end_time_xy - start_time_xy
                # print('normalizeVariance time: ' + str(time_dif_xy))

                bigSize = min(img.shape[0],img.shape[1])
                stride = int(bigSize/20)
                stepSize = int(bigSize/20)
                bbox = getAdaboost(img, weekClass, alpha, bigSize, stepSize, stride)
                print('face:'+str(len(bbox)))

                original = cv2.imread(data_path+'/'+imgName,cv2.IMREAD_COLOR)
                boxImg = drawRectangle(original, bbox)
                write_image(boxImg, './result/'+str(imgName))
                for i in range(len(bbox)):
                    ibox = [bbox[i][0][0],bbox[i][0][1],bbox[i][0][2],bbox[i][0][2]]
                    # print('ibox:'+str(ibox))
                    resulti = {"iname":imgName, "bbox":ibox}
                    result.append(resulti)
            if jj==30:
                break

        with open("result.json", "w") as fp:
                json.dump(result , fp) 






    
if __name__ == "__main__":
    main()