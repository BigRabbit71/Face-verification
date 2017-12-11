#!/usr/bin/env python2
#
# To get image's embedding
# By Xia
# 2017/08/23
#
# eg: ./Openface_getEmbedding.py /Users/xyq/PycharmProjects/faces/webface



import time

start = time.time()

import argparse
import cv2
import itertools
import os
import csv
from keras.models import *


import numpy as np
np.set_printoptions(precision=3)

import openface

modelDir = '/Users/xyq/openface/models'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
csvDir = '/Users/xyq/PycharmProjects/faces/CSVFiles/openfaceOut128.csv'

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep


def get128FeaCSV(csvDir):
    print args.imgs
    imgDir = str(args.imgs[0])

    if not os.path.exists(csvDir):
        print 'Start getting 128 features...'
        with open(csvDir, 'wb') as csvfile:
            fieldnames = ['imgPath', '128features', 'personID']
            # fieldnames = ['imgPath', '128features']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()

            for pathName, labelNames, imgNames in os.walk(imgDir):
                for labelName in labelNames:
                    imgPaths = os.path.join(pathName, labelName)
                    # print(imgPaths)
                    for imgName in os.listdir(imgPaths):
                        if imgName.endswith('.jpg'):
                            # print(imgName)
                            img = os.path.join(str(imgPaths), imgName)
                            print img
                            try:
                                rep = getRep(img)
                                # print rep.shape
                                # rep = np.array(rep)
                                writer.writerow({'imgPath': img, '128features': rep, 'personID': labelName})

                            except Exception, e:
                                # print e
                                # print str(e).split(" ")[0]
                                if(str(e).split(" ")[0] == "Unable"):
                                    print "Unable to find a face: {}, skip it.".format(img)
                                    pass
    else:
        print 'CSV file already exist.'



if __name__ == '__main__':
    get128FeaCSV(csvDir)


print 'Done!'
