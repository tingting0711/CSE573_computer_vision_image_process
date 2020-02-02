import sys
import os
import re

import cv2
import numpy as np
import math

import time
from datetime import timedelta
def normalize(img):
	''' normalize the image/patch, let the min of it become 0, and max become 255, in order to reduce the influence of illumination.
	'''
	min = 255
	max = 0
	for i, row in enumerate(img):
	    for j, num in enumerate(row):
	        if(img[i][j]<min):
	            min = img[i][j]
	        if(img[i][j]>max):
	            max = img[i][j]
	img = [list(row) for row in img]
	for i, row in enumerate(img):
	    for j, num in enumerate(row):
	        img[i][j] = (float(img[i][j]-min)/(max-min))*255
	return img
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
def my_drawMatches(img1, img2, kp1, kp2, matches):
    color = (200,200,200)
    pt1 = []
    pt2 = []
    (hA, wA) = img1.shape[:2]
    (hB, wB) = img2.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = img1
    vis[0:hB, wA:] = img2
    for i in range(len(matches)):
        pt1.append((kp1[matches[i].trainIdx][0],kp1[matches[i].trainIdx][1]))
        pt2.append((kp2[matches[i].queryIdx][0],kp2[matches[i].queryIdx][1]))
    for i in range(len(matches)):
        j = i
        vis = cv2.circle(vis, (int(pt1[j][0]),int(pt1[j][1])), 4, color, 2)
        vis = cv2.circle(vis, (int(pt2[j][0]+wA),int(pt2[j][1])), 4, color, 2)
        cv2.line(vis, (int(pt1[j][0]),int(pt1[j][1])),(int(pt2[j][0]+wA),int(pt2[j][1])), color[i], 2)
    return vis
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
    for i in range(len(image)):
        # sample:{'iname': '2002/08/31/big/img_18008', 'bbox': [-6, 1, 38, 53], 'num': 4}
        # data:['53.968100', '38.000000', '-1.494904', '31.598276', '55.596600', '1']
        # {'iname': '2002/08/31/big/img_18008', 'bbox': [12, 28, 38, 53], 'num': 4}
        img_path = './originalPics/'+str(image[i]['iname']+'.jpg')
        positive_path = 'data/positive/'
        negative_path = 'data/negative/'
        img = read_image(img_path)
        num = image[i]['num']
        offx = (image[i]['bbox'][3] - image[i]['bbox'][2])
        ymin = image[i]['bbox'][0]
        ymax = ymin+2*image[i]['bbox'][2]
        xmin = image[i]['bbox'][1]+int(1.5*offx)
        xmax = xmin+2*image[i]['bbox'][2]
        if xmin>=0 and xmax<=len(img) and ymin>=0 and ymax<=len(img[1]):   
            crop_img = crop(img, max(xmin,0), min(xmax,len(img)), max(ymin,0), min(ymax,len(img[1])))
            ii = "%04d" % po
            write_image(crop_img,'data/10/'+str(ii)+'.jpg')
            # norm1 = normalize(crop_img)
            # write_image(norm1,'data/11/'+str(ii)+'.jpg')
            resize_img = cv2.resize(np.array(crop_img),(24,24))
            # write_image(resize_img,'data/20/'+str(ii)+'.jpg')
            (mean , stddv) = cv2.meanStdDev(resize_img)
            norm = (resize_img-mean)/stddv
            norm = np.array(normalize(norm))
            cv2.imwrite(positive_path+str(ii)+'.jpg', norm)
            # write_image(norm,positive_path+str(ii)+'.jpg')
            po+=1
            ii = "%04d" % po
            flip = cv2.flip(norm,1)
            cv2.imwrite(positive_path+str(ii)+'.jpg', flip)
            po+=1
        # if po>=20:
        # 	print("po:"+str(po))
        # 	exit()
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
    for img_jpg in data_path:
        if len(images)>=10000:
            break
        if img_jpg.endswith(".jpg"):
            img = np.array(read_image(str(image_path)+str(img_jpg)))
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
            for w in range(1,width-x,2*increase):
                for h in range(1,height-y,increase):
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
            for w in range(1,width-x,increase):
                for h in range(1,height-y,2*increase):
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
# def constract_adaboost(feature,positive,negative,weight,false_positive_rate=0.3,true_positive_rate=0.99):
def constract_adaboost(feature,positive,negative,weight,cas_j,false_positive_rate=0.3,true_positive_rate=0.99):
    print("begin training adaboost!")
    # print("face images:"+str(len(positive))+", no-face images:"+str(len(negative))+", feature:"+str(len(feature)))
    weekClass = []
    alpha = []
    new_negative = []
    new_weight = []
    m = len(negative)
    l = len(positive)   
    # p_weight = (np.ones(l)*1.0)/(2*l)
    # n_weight = (np.ones(m)*1.0)/(2*m)
    # weight = np.hstack((p_weight,n_weight))
    # images = np.vstack((positive,negative))
    labels = np.hstack((np.ones(l),np.zeros(m)))
    i=0
    while True:
        # print('---------------adaboost'+str(i)+':')
        i+=1
        # print('weight:'+str(weight))
        weight = np.array(weight/weight.sum())
        images = np.vstack((positive,negative))
        bestFeature, weightErr, adabEst, bestErr= constract_best_feature(feature,images,labels,weight,cas_j,i)
        # print('feature:'+str(bestFeature[0])+', threshold:'+str(bestFeature[1])+', polarity:'+str(bestFeature[2])+', num of error:'+str(bestErr.sum())+',weightErr:'+str(weightErr))
        # print('num of error:'+str(bestErr.sum())+',weightErr:'+str(weightErr))
        expon = 1-bestErr
        beta = weightErr/(1.0-weightErr)
        alpha_ = np.log((1.0-weightErr)/max(weightErr,1e-16))
        alpha.append(alpha_)
        weight = np.multiply(np.power(beta,expon),weight)
        weekClass.append(bestFeature)
        (false_posi, true_posi, tmin) =calculate_ensemble_error(weekClass,alpha,positive,negative)
        false_posi_r = false_posi.sum()/m
        true_posi_r = true_posi.sum()/l
        print('---------------adaboost'+str(i)+':, fp_rate:'+str(false_posi_r)+',tp_rate:'+str(true_posi_r)+', tmin:'+str(tmin))
        if false_posi_r<=false_positive_rate:
            print('final have '+str(i)+' adaboost feature, fp_rate:'+str(false_posi_r)+',tp_rate:'+str(true_posi_r))
            new_weight = weight[0:l]
            for j in range(len(negative)):
                if false_posi[j]==1:
                    new_negative.append(negative[j])
                    new_weight = np.append(new_weight, [weight[j+l]], axis=0)
            new_negative = np.array(new_negative)
            new_weight = np.array(new_weight)
            print('new_negative:'+str(len(new_negative)))
            break
    # return (alpha,weekClass,weight,tmin,false_posi_r, true_posi_r, negative)
    return (alpha,weekClass,tmin,false_posi_r, true_posi_r, new_negative, new_weight)
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
    sort_pred_po = np.sort(agg_po)
    tmin = sort_pred_po[int(l/100)-1]
    true_positive = np.zeros(len(positive), dtype=int)
    false_positive = np.zeros(len(negative), dtype=int)
    # if cascade==False:
	   #  true_positive[agg_po>=0.5*np.array(alpha).sum()] = 1
	   #  false_positive[agg_ne>=0.5*np.array(alpha).sum()] = 1
	   #  return (false_positive.sum()/m, true_positive.sum()/l)
    true_positive[agg_po>=tmin] = 1
    false_positive[agg_ne>=tmin] = 1
    return (false_positive, true_positive, tmin)
def constract_best_feature(features,images,labels,weights,cas_j,index,step=5):
    adabEst = []
    weightErr = []
    bestErr = []
    if index==1:
        pred_v_all = []
        bestall = []
        print('new cascade, recompute new bestall, pred_v_all')
    else:
        pred_v_all = np.load('pred_v_all_'+str(cas_j)+'.npy')
        bestall = np.load('bestall_'+str(cas_j)+'.npy')
    # print("images:"+str(len(images))+", weights:"+str(weights))
    print('begin constract_best_feature:')
    minErr = float('inf')
    start_time_b = time.time()
    for i,feature in enumerate(features):
        if i%1000==0:
            print('begin find best result of '+str(i)+'-th/'+str(len(features)))
        if index==1:
            add = feature[0]
            sub = feature[1]
            pred_f = pred_feature(feature,images)
            pred_v_best = []
            minErr_i = float('inf')
            best_t = 0
            best_p = '>'
            for j in range(0,len(pred_f),step):
                threshold = pred_f[j]
                for polarity in ['<','>']:
                    pred_v = pred_class(pred_f,threshold,polarity)
                    error = np.ones(len(images))
                    error[pred_v==labels] = 0
                    weightErr = (weights*error).sum()
                    if weightErr<minErr_i:
                        pred_v_best= pred_v
                        best_t = threshold
                        best_p = polarity
                    if weightErr<minErr:
                        minErr = weightErr
                        best = {'feature':feature,'threshold':threshold, 'polarity':polarity}
                        bestFeature = np.array(list(best.values()))
                        ##bestErr if it predicted corectly, then 0; else 1.
                        bestErr = error
                        ##predict 0 indicate no-face; 1 indicate face.
                        adabEst= pred_v
            pred_v_all.append(pred_v_best)
            bestall.append([feature, best_t, best_p])
        else:
            pred_v = pred_v_all[i]
            error = np.ones(len(images))
            error[pred_v==labels] = 0
            weightErr = (weights*error).sum()
            if weightErr<minErr:
                minErr = weightErr
                best = {'feature':bestall[i][0],'threshold':bestall[i][1], 'polarity':bestall[i][2]}
                bestFeature = np.array(list(best.values()))
                ##bestErr if it predicted corectly, then 0; else 1.
                bestErr = error
                ##predict 0 indicate no-face; 1 indicate face.
                adabEst= pred_v
    end_time_b = time.time()
    time_dif_b = end_time_b - start_time_b
    print("find best feature in "+str(index)+": " + str(timedelta(seconds=int(round(time_dif_b)))))
    pred_v_all = np.array(pred_v_all)
    if index==1:
        np.save('pred_v_all_'+str(cas_j)+'.npy',pred_v_all)
        np.save('bestall_'+str(cas_j)+'.npy',bestall)

    return (bestFeature,minErr,adabEst,bestErr)
def pred_class(preds,threshold,polarity):
    result = np.ones(len(preds))
    if polarity == '<':
        result[preds[:]>=threshold] = 0
    else:
        result[preds[:]<=threshold] = 0
    return result
def pred_feature(feature,images,amplify=1):
    print('feature:'+str(type(feature)))
    print('feature:'+str(feature))
    print('images:'+str(type(images)))
    print(len(images))
    print(len(images[0]))
    print(len(images[0][0]))
    exit()
    add = feature[0]
    sub = feature[1]
    add_all = images[:,[amplify*x[0] for x in add],[amplify*y[1] for y in add]].sum(axis=-1)
    sub_all = images[:,[amplify*x[0] for x in sub],[amplify*y[1] for y in sub]].sum(axis=-1)
    result = add_all - sub_all
    print(type(feature))
    print(feature)
    exit()
    return result
def constract_cascade(feature,positive,negative,false_positive_rate,target_false_positive_rate=0.01):
    cascade = []
    m = len(negative)
    l = len(positive)
    p_weight = (np.ones(l)*1.0)/(2*l)
    n_weight = (np.ones(m)*1.0)/(2*m)
    weight = np.hstack((p_weight,n_weight))
    print("begin training cascade!")
    # print("face images:"+str(len(positive))+", no-face images:"+str(len(negative))+", feature:"+str(len(feature))+".")
    j=1
    pos_all_r = neg_all_r = 1
    while True:
        print('------------------stage'+str(j)+'------------------')
        print("face images:"+str(len(positive))+", no-face images:"+str(len(negative))+", feature:"+str(len(feature))+".")
        # (alpha,weekClass,weight,tmin,false_posi, true_posi,negative) = constract_adaboost(feature,positive,negative,weight,false_positive_rate)
        (alpha,weekClass,tmin,false_posi, true_posi,new_negative, new_weight) = constract_adaboost(feature,positive,negative,weight, j,false_positive_rate)
        negative = new_negative
        weight = new_weight
        print('weight:'+str(len(weight)))
        cascade.append((alpha,weekClass,tmin,false_posi, true_posi))
        pos_all_r *= true_posi
        neg_all_r *= false_posi
        print('pos_all_r:'+str(pos_all_r)+',neg_all_r:'+str(neg_all_r)+'.tmin:'+str(tmin))
        print('\n\n')
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
    x1,y1,w1,h1 = box1
    x2,y2,w2,h2 = box2
    s1 = (x1+w1)/2
    t1 = (y1+h1)/2
    s2 = (x2+w2)/2
    t2 = (y2+h2)/2

    if abs(s1-s2)*2<(w1+w2) and abs(y1-y2)*2<(h1+h2):
        return True
    else :
        return False

def get_test_data(data_path):
    #get all data(image) inside 'data_path'(including sub path, sub sub path and so on)
    data = []
    i = 0
    for root,dirs,files in os.walk(data_path,topdown=True):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith('.jpg'):
                i+=1
                img_path = str(os.path.join(root,file))
                img = read_image(img_path)
                data.append(img)
                if i%100==0:
                    print('find '+str(i)+' images')
    # print('get '+str(len(data))+' test_images from directory:'+str(data_path)+'.') #1507
    return data
def main():
    #python detect.py [datadirectory]
    
    ###### preprocess part, to get train image:positive + negative;
    # get_data('./FDDB-folds/')
    # result_array = np.load('face_list.npy')
    # result = result_array.tolist()
    # preprocess(result)  # create 24*24 positive/negative gray image
    ####### create 24*24 positive/negative integral image
    # positive = get_integral('./data/positive/')
    # np.save('positive_it.npy',positive)
    # negative = get_integral('./data/negative/')
    # np.save('negative_it.npy',negative)
    # exit()
    traning  = 1
    stride = 8
    increase = 8

    if traning==1:
        ####training cascaded adaboost feature whose all_false_positive_rate is less than 0.001
        # os.chdir("drive/cv_project3/")
        positive_it = np.load('positive_it.npy')
        negative_it = np.load('negative_it.npy')
        # print(len(positive_it))
        # exit()
        false_positive = 0.4
        feature = create_feature(24,24,stride,increase)
        print('stride:'+str(stride)+', increase:'+str(increase)+', feature:'+str(len(feature)))
        print('----------------------------------------\n')
        cascade = constract_cascade(feature,positive_it[0:100],negative_it[0:100],false_positive)
        cascade = np.array(cascade)
        np.save('cascade2.npy',cascade)
    else :
        ####testing 
        cascade = np.load('cascade.npy')
        data_path = str(sys.argv[1])
        # test_images = get_test_data(data_path) # for submit
        test_images = get_test_data1('./FDDB-data/')
        np.save('gray_test_images-1000.npy',test_images)
        print('save')
        exit()
        test_images = np.load('gray_test_images.npy')
        # exit()
        stride = 3
        bigSize = 12
        result = []

        for n in range(len(test_images)):
            iname = test_images[n][0].values()
            img = test_images[n][1].values()
            print('iname:'+str(iname))
            print('img:'+str(len(img)))
            exit()
            bbox = []
            small_l = min(img.shape[0],img.shape[0])
            for size in range(24,small_l,bigSize):
                for x in range(0,test_images[n].shape[0]-size,stride):
                    for y in range(0,test_images[n].shape[1]-size,stride):
                        patch = img[x:x+size,y:y+size]
                        integ_patch = integral_image(patch)
                        flag = cascade_predict(cascade,integ_patch)
                        if flag:
                            for ind,box in enumerate(bbox):
                                ibox = (x,y,size,size)
                                inte_flags = intersect_box(ibox,box)
                                if inte_flags==0:
                                    bbox.append(ibox)
                                    result_i = {'iname':iname,'bbox':ibox}
                                    result.append(result_i)
            berak




    
if __name__ == "__main__":
    main()