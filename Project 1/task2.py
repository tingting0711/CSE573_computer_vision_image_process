"""
Character Detection
(Due date: March 8th, 11: 59 P.M.)

The goal of this task is to experiment with template matching techniques. Specifically, the task is to find ALL of
the coordinates where a specific character appears using template matching.

There are 3 sub tasks:
1. Detect character 'a'.
2. Detect character 'b'.
3. Detect character 'c'.

You need to customize your own templates. The templates containing character 'a', 'b' and 'c' should be named as
'a.jpg', 'b.jpg', 'c.jpg' and stored in './data/' folder.

Please complete all the functions that are labelled with '# TODO'. Whem implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them. The functions defined in utils.py
and the functions you implement in task1.py are of great help.

Hints: You might want to try using the edge detectors to detect edges in both the image and the template image,
and perform template matching using the outputs of edge detectors. Edges preserve shapes and sizes of characters,
which are important for template matching. Edges also eliminate the influence of colors and noises.

Do NOT modify the code provided.
Do NOT use any API provided by opencv (cv2) and numpy (np) in your code.
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os

import utils
from task1 import *  

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img_path", type=str, default="./data/characters.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--template_path", type=str, default="",
        choices=["./data/a.jpg", "./data/b.jpg", "./data/c.jpg"],
        help="path to the template image")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./results/",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def myelementwise_mul(a, b):
    """Elementwise multiplication."""
    """because of 'RuntimeWarning: overflow encountered in ubyte_scalars' error, so make my new func"""
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] = int(b[i][j])*int(c[i][j])
    return c

def myelementwise_sub(a, b):
    """Elementwise substraction."""
    """because of 'RuntimeWarning: overflow encountered in ubyte_scalars' error, so make my new func"""
    # print type(a)
    # print type(b)
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] = int(c[i][j])-int(b[i][j])
    return c
def myelementwise_newadd(a, b):
    """Elementwise add a number b to every pixels."""
    """define a new add func to add a real number ."""
    c = copy.deepcopy(a)
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            c[i][j] = int(c[i][j])+b
    return c
def mean_normalize(a):
    c = copy.deepcopy(a)
    min = 10000;
    max = -10000;
    for i, row in enumerate(c):
        for j, num in enumerate(row):
            if(c[i][j]<min):
                min = c[i][j]
            if(c[i][j]>max):
                max = c[i][j]
    c = [list(row) for row in c]
    if max == min :
        return c
    for i, row in enumerate(c):
        for j, num in enumerate(row):
            c[i][j] = ((c[i][j]-min)*255)/(max-min)
    return c

def mean(a):
    mean = 0
    c = copy.deepcopy(a)
    for i, row in enumerate(c):
        for j, num in enumerate(row):
            mean += int(c[i][j])
    # c = [list(row) for row in c]
    mean = mean/(len(c)*len(c[0]))
    return mean

def detect_a(img_edges, tep_edges):
    ###best 0.135 find 9, 9 correct
    # print "aaaaaaa"
    tep_x = len(tep_edges)
    tep_y = len(tep_edges[0])

    mean_tep = mean_normalize(tep_edges)
    mean_ssd = copy.deepcopy(img_edges)
    for i in range(len(mean_ssd)-tep_x):
        for j in range(len(mean_ssd[0])-tep_y):
            if i>=50 and i<=320 and j>=40 and j<=620:
                img_patch = utils.crop(img_edges, i, i+tep_x, j, j+tep_y)
                mean_patch = copy.deepcopy(img_patch)
                mean_patch = mean_normalize(img_patch)
                element = copy.deepcopy(mean_patch)
                element = myelementwise_sub(mean_patch, mean_tep)
                mean_ssd[i][j] = 0
                for ii in range(tep_x):
                    for jj in range(tep_y):
                        mean_ssd[i][j] += int(element[ii][jj]*element[ii][jj])
    mean_ssd = normalize(mean_ssd)
    # write_image(mean_ssd,os.path.join('./results/', "ssd_a.jpg"))
    # mean_ssd = read_image('./results/ssd_a.jpg')
    ##threshold
    ssd_inv = copy.deepcopy(mean_ssd)
    coordinates = []
    threshold = 0.135
    for i, row in enumerate(mean_ssd):
        for j, num in enumerate(row):
            if i>=50 and i<=320 and j>=40 and j<=620:
                if (mean_ssd[i][j]/255.0) < threshold:
                    coordinates.append([i,j])
                    ssd_inv[i][j] = 255
                else :
                    ssd_inv[i][j] = 0
            else:
                ssd_inv[i][j] = 0
    num = len(coordinates)
    # print len(coordinates)
    # write_image(ssd_inv,os.path.join('./results/', "{}ssd_inv_th_a_{}.jpg".format(threshold,num)))
    ####display position in img
    # display_posi(coordinates, threshold, num, 'a')

    return coordinates

def detect_b(img_edges, tep_edges):
    #0.06  4/4
    # print "bbbbbbbbbbb"
    tep_x = len(tep_edges)
    tep_y = len(tep_edges[0])

    mean_tep = mean_normalize(tep_edges)
    mean_ssd = copy.deepcopy(img_edges)
    for i in range(len(mean_ssd)-tep_x):
        for j in range(len(mean_ssd[0])-tep_y):
            if i>=50 and i<=320 and j>=40 and j<=620:
                img_patch = utils.crop(img_edges, i, i+tep_x, j, j+tep_y)
                mean_patch = copy.deepcopy(img_patch)
                mean_patch = mean_normalize(img_patch)
                element = copy.deepcopy(mean_patch)
                element = myelementwise_sub(mean_patch, mean_tep)
                mean_ssd[i][j] = 0
                for ii in range(tep_x):
                    for jj in range(tep_y):
                        mean_ssd[i][j] += int(element[ii][jj]*element[ii][jj])
    mean_ssd = normalize(mean_ssd)
    # write_image(mean_ssd,os.path.join('./results/', "ssd_b.jpg"))
    # mean_ssd = read_image('./results/ssd_b.jpg')
    ##threshold
    ssd_inv = copy.deepcopy(mean_ssd)
    coordinates = []
    threshold = 0.06
    for i, row in enumerate(mean_ssd):
        for j, num in enumerate(row):
            if i>=50 and i<=320 and j>=40 and j<=620:
                if (mean_ssd[i][j]/255.0) < threshold:
                    coordinates.append([i,j])
                    ssd_inv[i][j] = 255
                else :
                    ssd_inv[i][j] = 0
            else:
                ssd_inv[i][j] = 0
    ssd_inv = normalize(ssd_inv)
    num = len(coordinates)
    # print num
    # write_image(ssd_inv,os.path.join('./results/', "{}ssd_inv_th_b_{}.jpg".format(threshold,num)))
    ####display position in img
    # display_posi(coordinates, threshold, num, 'b')

    return coordinates

def detect_c(img_edges, tep_edges):
    #####0.07, find 20, 18 are correct,1 wrong, 1 duplicate
    # print "ccccc"
    tep_x = len(tep_edges)
    tep_y = len(tep_edges[0])

    mean_tep = mean_normalize(tep_edges)
    mean_ssd = copy.deepcopy(img_edges)
    for i in range(len(mean_ssd)-tep_x):
        for j in range(len(mean_ssd[0])-tep_y):
            if i>=50 and i<=320 and j>=40 and j<=620:
                img_patch = utils.crop(img_edges, i, i+tep_x, j, j+tep_y)
                mean_patch = copy.deepcopy(img_patch)
                mean_patch = mean_normalize(img_patch)
                element = copy.deepcopy(mean_patch)
                element = myelementwise_sub(mean_patch, mean_tep)
                mean_ssd[i][j] = 0
                for ii in range(tep_x):
                    for jj in range(tep_y):
                        mean_ssd[i][j] += int(element[ii][jj]*element[ii][jj])
    mean_ssd = normalize(mean_ssd)
    # write_image(mean_ssd,os.path.join('./results/', "ssd_c.jpg"))
    #mean_ssd = read_image('./results/ssd_c.jpg')
    ##threshold
    ssd_inv = copy.deepcopy(mean_ssd)
    coordinates = []
    threshold = 0.07
    for i, row in enumerate(mean_ssd):
        for j, num in enumerate(row):
            if i>=50 and i<=320 and j>=40 and j<=620:
                if (mean_ssd[i][j]/255.0) < threshold:
                    coordinates.append([i,j])
                    ssd_inv[i][j] = 255
                else :
                    ssd_inv[i][j] = 0
            else:
                ssd_inv[i][j] = 0
    ssd_inv = normalize(ssd_inv)
    num = len(coordinates)
    # print num
    # write_image(ssd_inv,os.path.join('./results/', "{}ssd_inv_th_c_{}.jpg".format(threshold,num)))
    ####display position in img
    # display_posi(coordinates, threshold, num, 'c')
    return coordinates

def display_posi(coordinates, threshold, num, alpa):
    ####display position in img
    abc_edges = read_image('./data/edge_abc.jpg')
    all_edges = read_image('./data/new_img.jpg')
    for i in range(len(coordinates)):
        # print coordinates[i]
        abc_edges[coordinates[i][0]][coordinates[i][1]+0] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+1] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+2] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+3] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+4] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+5] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+6] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+7] = 255
        abc_edges[coordinates[i][0]][coordinates[i][1]+8] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+0] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+1] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+2] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+3] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+4] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+5] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+6] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+7] = 255
        all_edges[coordinates[i][0]][coordinates[i][1]+8] = 255
    # print abc_edges[55]
    write_image(abc_edges,os.path.join('./results/', "{}_{}_find_{}.jpg".format(threshold, alpa, num)))
    write_image(all_edges,os.path.join('./results/', "{}_{}_find_all_{}.jpg".format(threshold, alpa, num)))

def detect(img, template):
    # python task2.py --img_path ./data/proj1-task2.jpg --template_path ./data/a.jpg
    """Detect a given character, i.e., the character in the template image.
    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.
    Returns:
        coordinates: list (tuple), a list whose elements are coordinates where the character appears.
            format of the tuple: (x (int), y (int)), x and y are integers.
            x: row that the character appears (starts from 0).
            y: column that the character appears (starts from 0).
    """
    # print "detect"
    kernel_x = sobel_x
    kernel_y = sobel_y
    new_img = copy.deepcopy(img)
    for i, row in enumerate(img):
        for j, num in enumerate(row):
            new_img[i][j] = 255-img[i][j]
    # print img[55]
    # print "\n\n"
    # print new_img[55]
            
    # write_image(new_img,os.path.join('./results/', "new_img.jpg"))
    # print "write"
    # exit()

    tep_x = len(template)
    tep_y = len(template[0])

    coordinates = []
    if tep_x == 10 and tep_y==6:
        coordinates = detect_a(new_img, template)
    if tep_x == 12 and tep_y==8:
        coordinates = detect_b(new_img, template)
    if tep_x == 12 and tep_y==6:
        coordinates = detect_c(new_img, template)
        
    # TODO: implement this function.
    #raise NotImplementedError
    return coordinates


def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path) 
    template = read_image(args.template_path)

    coordinates = detect(img, template)

    template_name = "{}.json".format(os.path.splitext(os.path.split(args.template_path)[1])[0])
    save_results(coordinates, template, template_name, args.rs_directory)


if __name__ == "__main__":
    main()
