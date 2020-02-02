import copy
import sys
import os
import random

import cv2
import numpy as np

# import utils

class MyMatch():
    """docstring for MyMatch"""
    distance = -1
    trainIdx = -1
    queryIdx = -1

class MyRansac():
    """for computing ransac"""
    gap = 0
    count = 0


def my_show(img):
    resize_img = cv2.resize(img,(int(img.shape[1]/6),int(img.shape[0]/6)),cv2.INTER_LINEAR)
    cv2.imshow("img", resize_img)
    k = cv2.waitKey(0)
    if k==27:
        cv2.destroyAllWindows() 

def my_show1(img):
    cv2.imshow("img", img)
    k = cv2.waitKey(0)
    if k==27:
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

def my_drawMatches(img1, img2, kp1, kp2, matches,color):
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
        vis = cv2.circle(vis, (int(pt1[j][0]),int(pt1[j][1])), 4, color[i], 2)
        vis = cv2.circle(vis, (int(pt2[j][0]+wA),int(pt2[j][1])), 4, color[i], 2)
        cv2.line(vis, (int(pt1[j][0]),int(pt1[j][1])),(int(pt2[j][0]+wA),int(pt2[j][1])), color[i], 2)
    return vis


def new_drawMatches(img1, img2, left_ps0, right_ps0):
    color = []
    for i in range(len(left_ps0)):
        color.append((int(random.random()*255),int(random.random()*255),int(random.random()*255)))

    (hA, wA) = img1.shape[:2]
    (hB, wB) = img2.shape[:2]
    vis0 = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis0[0:hA, 0:wA] = img1
    vis0[0:hB, wA:] = img2
    for i in range(len(left_ps0)):
        j = i
        vis0 = cv2.circle(vis0, (int(left_ps0[j][0]),int(left_ps0[j][1])), 4, color[i], 2)
        vis0 = cv2.circle(vis0, (int(right_ps0[j][0]+wA),int(right_ps0[j][1])), 4, color[i], 2)
        cv2.line(vis0, (int(left_ps0[j][0]),int(left_ps0[j][1])),(int(right_ps0[j][0]+wA),int(right_ps0[j][1])), color[i], 2)
    vis0 = my_resize(vis0)
    return vis0

def my_drawKeypoints(img1,pt):
    color = []
    for i in range(len(pt)):
        color.append((int(random.random()*255),int(random.random()*255),int(random.random()*255)))
    vis = copy.deepcopy(img1)
    for i in range(len(pt)):
        vis = cv2.circle(vis, (int(pt[i][0]),int(pt[i][1])), 4, color[i], 2)
    return vis
def my_draw(img1,pt,color_i,color):
    pt = (int(pt[0]),int(pt[1]))
    vis = cv2.circle(img1, pt, 3, color[color_i], 1)
    return vis

def my_match(kp1,des1,kp2,des2):
    matches = []
    for i in range(len(des1)):
        mm= MyMatch()
        mm.distance = 10000
        mm.trainIdx = i
        for j in range(len(des2)):
            distance = np.linalg.norm(des1[i] - des2[j])
            if distance < mm.distance:
                mm.distance = distance
                mm.queryIdx = j
        flag = 1
        for k in range(len(des1)):
            distance = np.linalg.norm(des2[mm.queryIdx]-des1[k])
            if distance < mm.distance:
                flag = 0
        if flag == 1:
            matches.append(mm)
    matches = sorted(matches, key=lambda x:x.distance)
    if len(matches)<4:
        print('too less match point!')
    # print('len(matches):'+str(len(matches)))
    # print(len(matches))
    # goodmatches = []
    # min_dist = 1000
    # max_dist = 0
    # for i in range(len(matches)):
    #     if matches[i].distance > max_dist:
    #         max_dist = matches[i].distance
    #     if matches[i].distance < min_dist:
    #         min_dist = matches[i].distance
    # print('min_dist:'+str(min_dist))
    # print('max_dist:'+str(max_dist))

    # for i in range(len(matches)):
    #     # matches[i].distance <= 2*min_dist or 
    #     if matches[i].distance <= min_dist+0.2*(max_dist-min_dist):
    #         goodmatches.append(matches[i])
    # print('len(goodmatches):'+str(len(goodmatches)))
    # if len(matches)>100:
    #     matches = matches[0:100]
    return matches

def find_position(img1,img2,kp1,kp2,matches1,matches2):
    p1 = 0
    p2 = 0
    (h1, w1) = img1.shape[:2]
    (h2, w2) = img2.shape[:2]
    for i in range(len(matches1)):
        if kp1[matches1[i].trainIdx][0]>w1/2:
            p1+= 1
    for i in range(len(matches2)):
        if kp2[matches2[i].queryIdx][0]>w2/2:
            p2+= 1
    if p1 >= p2:
        return(img1,img2,0)
    else:
        return(img2,img1,1)
       
def find_position_2(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create(500)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp1,des1 = sift.detectAndCompute(gray1,None)
    kp2,des2 = sift.detectAndCompute(gray2,None)
    kp11 = np.float32([kp.pt for kp in kp1])
    kp22 = np.float32([kp.pt for kp in kp2])

    # find left right
    matches_1_2 = my_match(kp11,des1,kp22,des2)
    (left,right,flags)=find_position(img1,img2,kp11,kp22,matches_1_2,matches_1_2)

    return (left,right,flags)

def find_position_3(img1,img2,img3):
    sift = cv2.xfeatures2d.SIFT_create(500)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    kp1,des1 = sift.detectAndCompute(gray1,None)
    kp2,des2 = sift.detectAndCompute(gray2,None)
    kp3,des3 = sift.detectAndCompute(gray3,None)
    kp11 = np.float32([kp.pt for kp in kp1])
    kp22 = np.float32([kp.pt for kp in kp2])
    kp33 = np.float32([kp.pt for kp in kp3])
    # img1 = my_drawKeypoints(img1,kp11)
    # img2 = my_drawKeypoints(img2,kp22)
    # img3 = my_drawKeypoints(img3,kp33)
    # write_image(img1, '../process/keypint1.jpg')
    # write_image(img2, '../process/keypint2.jpg')
    # write_image(img3, '../process/keypint3.jpg')
    # exit()
    p1 = 0
    p2 = 0
    p3 = 0

    # find left middle right
    print('begin finding left middle right')
    print('begin find position:matches_1_2')
    matches_1_2 = my_match(kp11,des1,kp22,des2)
    # print('my_match1-2:'+str(len(matches_1_2)))
    (left1_2,right1_2,flags1_2)=find_position(img1,img2,kp11,kp22,matches_1_2,matches_1_2)
    if flags1_2:
        p1 +=1
    else:
        p2 +=1
    print('begin find position:matches_1_3')
    matches_1_3 = my_match(kp11,des1,kp33,des3)
    # print('my_match1-3:'+str(len(matches_1_3)))
    (left1_3,right1_3,flags1_3)=find_position(img1,img3,kp11,kp33,matches_1_3,matches_1_3)
    if flags1_3:
        p1 +=1
    else:
        p3 +=1
    print('begin find position:matches_2_3')
    matches_2_3 = my_match(kp22,des2,kp33,des3)
    # print('my_match2-3:'+str(len(matches_2_3)))
    (left2_3,right2_3,flags2_3)=find_position(img2,img3,kp22,kp33,matches_2_3,matches_2_3)
    if flags2_3:
        p2 +=1
    else:
        p3 +=1

    # print('p1:'+str(p1)+';p2:'+str(p2)+';p3:'+str(p3))

    # --------------------------
    if p1==0:
        left = img1
        if  flags2_3==0 and p3==2:
            middle = img2
            right = img3
        elif flags2_3==1 and p2==2:
            middle = img3
            right = img2

    elif p1==1:
        middle = img1
        if flags1_2==1 and flags1_3==0:
            right = img3
            left = img2
        elif flags1_2==0 and flags1_3==1:
            right = img2
            left = img3

    elif p1 == 2:
        right = img1
        if  flags2_3==0 and p3==1:
            left = img2
            middle = img3
        elif flags2_3==1 and p2==1:
            left = img3
            middle = img2

    return (left,middle, right)
def takeCount(elem):
    return elem.count
def uniformChoise(left_ps0,right_ps0):
    left = []
    right = []
    length = len(left_ps0)
    left.append(left_ps0[0])
    left.append(left_ps0[int(length/4)])
    left.append(left_ps0[int(length/2)])
    left.append(left_ps0[length-1])

    right.append(right_ps0[0])
    right.append(right_ps0[int(length/4)])
    right.append(right_ps0[int(length/2)])
    right.append(right_ps0[length-1])

    left = np.array(left)
    right = np.array(right)
    return (left,right)
def firstChoise(left_ps0,right_ps0):
    left = left_ps0[0:4]
    right = right_ps0[0:4]
    return (left,right)
def ransacChoise(left_ps0,right_ps0,inverse=0):
    minDist = 10000
    f_left = []
    f_right = []
    number = 0
    # left_ps0 = np.array(left_ps0, dtype = "float32")
    # right_ps0 = np.array(right_ps0, dtype = "float32")
    left_ps0 = [[list(row)] for row in left_ps0]
    left_ps0 = np.array(left_ps0)
    right_ps0 = [[list(row)] for row in right_ps0]
    right_ps0 = np.array(right_ps0)
    for i in range(len(left_ps0)):
        if number>=100000:
            break;
        for j in range(i+1,len(left_ps0)):
            if number>=100000:
                break;
            for k in range(j+1,len(left_ps0)):
                if number>=100000:
                    break;
                for q in range(k+1,len(left_ps0)):
                    if number>=100000:
                        break;
                    left = []
                    right = []
                    left.append(left_ps0[i])
                    left.append(left_ps0[j])
                    left.append(left_ps0[k])
                    left.append(left_ps0[q])
                    right.append(right_ps0[i])
                    right.append(right_ps0[j])
                    right.append(right_ps0[k])
                    right.append(right_ps0[q])
                    left = np.array(left, dtype = "float32")
                    right = np.array(right, dtype = "float32")
                    if inverse==0:
                        H = cv2.getPerspectiveTransform(right,left)
                        e_left = cv2.perspectiveTransform(right_ps0,H)
                        # print("e_left:"+str(e_left))
                        # print("left_ps0:"+str(left_ps0))
                        dis = np.linalg.norm(left_ps0-e_left)
                        if dis < minDist:
                            minDist = dis
                            f_left = left
                            f_right = right
                    else:
                        H = cv2.getPerspectiveTransform(left,right)
                        e_right = cv2.perspectiveTransform(left_ps0,H)
                        # print("e_right:"+str(e_right))
                        # print("right_ps0:"+str(right_ps0))
                        dis = np.linalg.norm(right_ps0-e_right)
                        if dis < minDist:
                            minDist = dis
                            f_left = left
                            f_right = right

                    number +=1
    f_left = np.squeeze(f_left)
    f_right = np.squeeze(f_right)
    print("iteration_number:"+str(number))
    print("f_left:"+str(f_left))
    print("f_right:"+str(f_right))
    # exit()
    return (f_left,f_right)
def my_ransac(left_ps,right_ps,matches,img1,img2,imgname,inverse=0):
    (hA, wA) = img1.shape[:2]
    (hB, wB) = img2.shape[:2]
    goodmatch = []
    xgap=[]
    ygap=[]
    x = []
    y = []
    for i in range(len(left_ps)):
        gap_x = (wA - left_ps[i][0]) + (right_ps[i][0])
        xgap.append(gap_x)
        flag = 0
        for j in range(len(x)):
            if xgap[i] == x[j].gap:
                x[j].count +=1
                flag = 1
        if flag == 0:
            xx = MyRansac()
            xx.gap = xgap[i]
            xx.count +=1
            x.append(xx)
        ###compute y
        gap_y = left_ps[i][1] - right_ps[i][1]
        ygap.append(gap_y)
        flag = 0
        for j in range(len(y)):
            if ygap[i] == y[j].gap:
                y[j].count +=1
                flag = 1
        if flag == 0:
            yy = MyRansac()
            yy.gap = ygap[i]
            yy.count +=1
            y.append(yy)

    ##delete disfferent xgap
    x.sort(key=takeCount,reverse=True)
    y.sort(key=takeCount,reverse=True)
    left_ps = left_ps.tolist()
    right_ps = right_ps.tolist()
    xmuch = x[0].gap
    ymuch = y[0].gap
    length = len(left_ps)
    i = 0
    t = 10
    while i < (len(xgap)):
        if xgap[i] > xmuch+t or xgap[i] <xmuch-t or ygap[i] > ymuch+t or ygap[i] <ymuch-t:
            left_ps.pop(i)
            right_ps.pop(i)
            matches.pop(i)
            xgap.pop(i)
            ygap.pop(i)
            i -=1
        i +=1
    vis0 = new_drawMatches(img1, img2, left_ps, right_ps)
    vis0 = my_resize(vis0)
    write_image(vis0, "../process/{}.jpg".format(imgname))
    left_ps = np.array(left_ps)
    right_ps = np.array(right_ps)
    if len(left_ps) < 4:
        print('too less match points')
    # (left,right) = uniformChoise(left_ps,right_ps)
    # (left,right) = firstChoise(left_ps,right_ps)
    (left,right) = ransacChoise(left_ps,right_ps,inverse)
    return (left,right)
   
def my_perspective(left,right,kp11,kp22,matches,imgname,inverse=0):
    pt1 = []
    pt2 = []
    for i in range(len(matches)):
        pt1.append([kp11[matches[i].trainIdx][0],kp11[matches[i].trainIdx][1]])
        pt2.append([kp22[matches[i].queryIdx][0],kp22[matches[i].queryIdx][1]])
        left_ps = np.array(pt1, dtype = "float32")#
        right_ps = np.array(pt2, dtype = "float32")#
    (left_p,right_p) = my_ransac(left_ps,right_ps,matches,left,right,imgname,inverse)
    left_p = np.array(left_p, dtype = "float32")
    right_p = np.array(right_p, dtype = "float32")
    vis0 = new_drawMatches(left, right, left_p, right_p)
    vis0 = my_resize(vis0)
    write_image(vis0, "../process/homo_{}.jpg".format(imgname))
    if inverse==0:
        h1,w1 = left.shape[:2]
        h2,w2 = right.shape[:2]
        pts1 = np.array(([[0,0],[0,h1],[w1,h1],[w1,0]]), dtype="float32").reshape(-1,1,2)
        pts2 = np.array(([[0,0],[0,h2],[w2,h2],[w2,0]]), dtype="float32").reshape(-1,1,2)
        H = cv2.getPerspectiveTransform(right_p,left_p)
        
        pts2_ = cv2.perspectiveTransform(pts2, H)
        # print("pts2:"+str(type(pts2)))
        # print("pts2:"+str(pts2))
        # print("H:"+str(type(H)))
        # print("H:"+str(H))
        # print("pts2_:"+str(type(pts2_)))
        # print("pts2_:"+str(pts2_))
        # exit()
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        result = cv2.warpPerspective(right, Ht.dot(H), (xmax-xmin, ymax-ymin))
        # write_image(result, "../process/crapright.jpg")
        # exit()
        # result = cv2.warpPerspective(right, np.array(H), (left.shape[1]+right.shape[1], max(right.shape[0],left.shape[0])))  
    elif inverse==1:
        h1,w1 = right.shape[:2]
        h2,w2 = left.shape[:2]
        pts1 = np.array(([[0,0],[0,h1],[w1,h1],[w1,0]]), dtype="float32").reshape(-1,1,2)
        pts2 = np.array(([[0,0],[0,h2],[w2,h2],[w2,0]]), dtype="float32").reshape(-1,1,2)
        H = cv2.getPerspectiveTransform(left_p,right_p)
        # result = cv2.warpPerspective(left, np.array(H), (left.shape[1]+right.shape[1], max(right.shape[0],left.shape[0])))  
        # write_image(result, "../process/wrapleft.jpg")
        pts2_ = cv2.perspectiveTransform(pts2, H)
        pts = np.concatenate((pts1, pts2_), axis=0)
        [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin,-ymin]
        Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
        result = cv2.warpPerspective(left, Ht.dot(H), (xmax-xmin, ymax-ymin))
        # write_image(result, "../process/crapleft.jpg")
        # exit()
        # H = cv2.getPerspectiveTransform(left_p,right_p)
        # result = cv2.warpPerspective(left, np.array(H), (left.shape[1]+right.shape[1], max(right.shape[0],left.shape[0])))  
    return (result,t)


def my_stitch(left,right,t,inversion=0):
    (h1, w1) = left.shape[:2]
    (h2, w2) = right.shape[:2]
    vis = np.zeros((h1+h2, w1+w2, 3), dtype="uint8")
    if inversion==0:
        vis[t[1]:h1+t[1], t[0]:w1+t[0]] = left
        left = copy.deepcopy(vis)
        for i in range(right.shape[1]):
            if right[:,i].any() and left[:,i].any():
                xmin = i
                break
        left1 = my_resize(left)
        xmax = left1.shape[1]
        print('xmin:'+str(xmin))
        print('xmax:'+str(xmax))
        print('left,size :'+str(left.shape))
        print('right,size :'+str(right.shape))
        for i in range(right.shape[0]):
            for j in range(right.shape[1]):
                if not right[i,j].any():
                    right[i,j] = left[i,j]
                elif not left[i,j].any():
                    right[i,j] = right[i,j]
                else:
                    alpha = float(j-xmin)/float(xmax-xmin)
                    if alpha<0 or alpha>1:
                        print(alpha)
                    right[i,j] = alpha*right[i,j] + (1-alpha)*left[i,j]
        return right
                    
    else :
        vis[t[1]:h2+t[1], t[0]:w2+t[0]] = right
        right = copy.deepcopy(vis)
        for i in range(left.shape[1]):
            if left[:,i].any() and right[:,i].any():
                xmin = i
                break
        left1 = my_resize(left)
        xmax = left1.shape[1]
        print('xmin:'+str(xmin))
        print('xmax:'+str(xmax))
        print('left,size :'+str(left.shape))
        print('right,size :'+str(right.shape))

        for i in range(left.shape[0]):
            for j in range(left.shape[1]):
                if not right[i,j].any():
                    left[i,j] = left[i,j]
                elif not left[i,j].any():
                    left[i,j] = right[i,j]
                else:
                    alpha = float(j-xmin)/float(xmax-xmin)
                    if alpha<0 or alpha>1:
                        print(alpha)
                    left[i,j] = alpha*right[i,j] + (1-alpha)*left[i,j]
        return left


def my_resize(result):
    for i in range(result.shape[1]):
        if result[:,i].any():
            xmin = i
            break
    for i in range(result.shape[1]-1, 0, -1):
        if result[:,i].any():
            xmax = i
            break
    for j in range(result.shape[0]):
        if result[j,:].any():
            ymin = j
            break
    for j in range(result.shape[0]-1, 0, -1):
        if result[j:,].any():
            ymax = j
            break

    # print('left-top:('+str(xmin)+','+str(ymin)+')')
    # print('right-bot:('+str(xmax)+','+str(ymax)+')')
    result = result[ymin:ymax,xmin:xmax]
    return result

def my_resize_x(result):
    for i in range(result.shape[1]-1, 0, -1):
        if result[:,i].any():
            xmax = i
            break
    result = result[:,0:xmax]
    return result

# def my_resize_img(img,imgname):
#     # #resive reduce resolution
#         # img1 = cv2.resize(img1,(int(0.5*img1.shape[1]),int(0.5*img1.shape[0])))
#         # img2 = cv2.resize(img2,(int(0.5*img2.shape[1]),int(0.5*img2.shape[0])))
#         # img3 = cv2.resize(img3,(int(0.5*img3.shape[1]),int(0.5*img3.shape[0])))
#         # write_image(img1, '../ubdata/img1.jpg')
#         # write_image(img2, '../ubdata/img2.jpg')
#         # write_image(img3, '../ubdata/img3.jpg')
#         # exit()
#     pass
def main():
    #python stitch.py [datadirectory]

    #read all images from data_directory
    data_directory = str('../'+sys.argv[1])
    data_path = os.listdir(data_directory)
    img_path = []
    for imgname in data_path:
        if imgname.endswith(".jpg") or imgname.endswith(".png") or imgname.endswith(".jpeg") or imgname.endswith(".bmp"):
            img_path.append(str(data_directory+'/'+imgname))
            #print(imgname)


    color = []
    for i in range(1000):
        color.append((int(random.random()*255),int(random.random()*255),int(random.random()*255)))
        
    num = len(img_path)
    if(num==1):
        #1 picture
        
        img1 = cv2.imread(img_path[0])
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        write_image(img1, "{}/panorama.jpg".format(data_directory))

    if(num==2):
        #2 pictures
        sift = cv2.xfeatures2d.SIFT_create()
        img1 = cv2.imread(img_path[0])
        img2 = cv2.imread(img_path[1])
        # width = int(img1.shape[1]/5)
        top1, bot1, left1, right1 = 100, 100, 0, 0
    
        # # # ##version 1:right->left
        # print("finding position of 2 images!")
        # (left,right,flags)=find_position_2(img1,img2)
        # write_image(left, '../process/1left.jpg')
        # write_image(right, '../process/2right.jpg')
        # print("computing keypoint")
        # kp1,des1 = sift.detectAndCompute(left,None)
        # kp2,des2 = sift.detectAndCompute(right,None)
        # kp11 = np.float32([kp.pt for kp in kp1])
        # kp22 = np.float32([kp.pt for kp in kp2])
        # print("drawing keypoint")
        # keypoints1 = my_drawKeypoints(img1,kp11)
        # write_image(keypoints1, '../process/keypoint_l.jpg')
        # keypoints2 = my_drawKeypoints(img2,kp22)
        # write_image(keypoints2, '../process/keypoint_r.jpg')

        # matches = my_match(kp11,des1,kp22,des2)
        # match_draw1 = my_drawMatches(left, right, kp11, kp22, matches,color)
        # write_image(match_draw1, '../process/3ori_match.jpg')
        # (right,t) = my_perspective(left,right,kp11,kp22,matches,'4ransac_left_m_right',0)
        # write_image(right, "../process/5left_w.jpg")
        # result = my_stitch(left,right,t,0)
        # write_image(result, "../process/6final.jpg")
        # # write_image(result, "{}/panorama.jpg".format(data_directory))
        # exit()
       

        # ##version 2:left->right
        ##beter result
        print("finding position of 2 images!")
        (left,right,flags)=find_position_2(img1,img2)
        write_image(left, '../process/1left.jpg')
        write_image(right, '../process/2right.jpg')
        print("computing keypoint")
        kp1,des1 = sift.detectAndCompute(left,None)
        kp2,des2 = sift.detectAndCompute(right,None)
        kp11 = np.float32([kp.pt for kp in kp1])
        kp22 = np.float32([kp.pt for kp in kp2])
        print("drawing keypoint")
        keypoints1 = my_drawKeypoints(img1,kp11)
        write_image(keypoints1, '../process/1keypoint_l.jpg')
        keypoints2 = my_drawKeypoints(img2,kp22)
        write_image(keypoints2, '../process/2keypoint_r.jpg')

        matches = my_match(kp11,des1,kp22,des2)
        match_draw1 = my_drawMatches(left, right, kp11, kp22, matches,color)

        write_image(match_draw1, '../process/3match_draw.jpg')
        
        (left,t) = my_perspective(left,right,kp11,kp22,matches,'4ransac_left_m_right',1)
        write_image(left, "../process/5left_w.jpg")
        result = my_stitch(left,right,t,1)
        write_image(result, "../process/6final.jpg")
        write_image(result, "{}/panorama.jpg".format(data_directory))

        
    if(num==3):
        #3 pictures
        sift = cv2.xfeatures2d.SIFT_create(500)
        img1 = cv2.imread(img_path[0])
        img2 = cv2.imread(img_path[1])
        img3 = cv2.imread(img_path[2])
        top1, bot1, left1, right1 = 100, 100, 0, 0
        match_1 = 0
        match_2 = 0
        match_3 = 0

        (left,middle,right)=find_position_3(img1,img2,img3)
        print('find left right middle')
    
        write_image(left, '../process/1left.jpg')
        write_image(middle, '../process/2middle.jpg')
        write_image(right, '../process/3right.jpg')

        
        kp1,des1 = sift.detectAndCompute(left,None)
        kp2,des2 = sift.detectAndCompute(middle,None)
        kp3,des3 = sift.detectAndCompute(right,None)
        kp11 = np.float32([kp.pt for kp in kp1])
        kp22 = np.float32([kp.pt for kp in kp2])
        kp33 = np.float32([kp.pt for kp in kp3])
        print("drawing keypoint")
        keypoints1 = my_drawKeypoints(left,kp11)
        write_image(keypoints1, '../process/1keypoint_l.jpg')
        keypoints2 = my_drawKeypoints(middle,kp22)
        write_image(keypoints2, '../process/2keypoint_m.jpg')
        keypoints3 = my_drawKeypoints(right,kp33)
        write_image(keypoints3, '../process/3keypoint_r.jpg')



        #compute and matches_m_r
        matches_m_r = my_match(kp22,des2,kp33,des3)
        match_draw2 = my_drawMatches(middle, right, kp22, kp33, matches_m_r,color)
        write_image(match_draw2, '../process/4ori_match_m_r.jpg')

        #stitich middle and right
        # my_perspective(left,right,kp11,kp22,matches,imgname,inverse=0):
        (result_r,t) = my_perspective(middle,right,kp22,kp33,matches_m_r,'5ransac_middle_m_right',0)
        write_image(result_r, "../process/6per_right.jpg")
        result_r = my_stitch(middle,result_r,t,0)
        write_image(result_r, "../process/7final_r.jpg")
        

        #compute and matches_l_m
        # result_r = cv2.copyMakeBorder(result_r, 0, 0, int(left.shape[1]), 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        write_image(result_r, '../process/8_pading_m_r.jpg')
        kp2,des2 = sift.detectAndCompute(result_r,None)
        kp22 = np.float32([kp.pt for kp in kp2])
        matches_l_m = my_match(kp11,des1,kp22,des2)
        match_draw1 = my_drawMatches(left, result_r, kp11, kp22, matches_l_m,color)
        write_image(match_draw1, '../process/9ori_match_l_m.jpg')
       
        #stitich left and result_r
        # my_perspective(left,right,kp11,kp22,matches,inverse=0):
        (left,t) = my_perspective(left,result_r,kp11,kp22,matches_l_m,'10ransac_left_m_mright',1)
        write_image(left, "../finprocessal/11per_left.jpg")
        
        result = my_stitch(left,result_r,t,1)
        write_image(result, "../process/12final.jpg")

        write_image(result, "{}/panorama.jpg".format(data_directory))
    if (num>=4):
        print("more than 3 images!")


    
if __name__ == "__main__":
    main()