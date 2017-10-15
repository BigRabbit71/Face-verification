#!/usr/bin/env python2
#
# Example to compare the faces in two images based on face's embedding.
# By XYQ
# 2017/08/22


import cv2
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.cross_validation import train_test_split

import numpy as np
np.set_printoptions(precision=3)

import openface
modelDir = '/home/xyq/openface/models'
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

img1Dir = '/home/xyq/PycharmProjects/Openface/006.jpg'
img2Dir = '/home/xyq/PycharmProjects/Openface/009.jpg'
modelDir = '/home/xyq/PycharmProjects/Openface/openface_train_model.h5'

align = openface.AlignDlib(os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
net = openface.TorchNeuralNet(os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))

def getRep(imgPath):
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        raise Exception("Unable to find a face: {}".format(imgPath))
    alignedFace = align.align(96, rgbImg, bb,
                              landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    rep = net.forward(alignedFace)

    return rep


def FaceVerifacation(img1Dir, img2Dir):
    try:
        rep1 = getRep(img1Dir)
        rep2 = getRep(img2Dir)
        featurePair = []
        featurePair.append(np.append(rep1, rep2, axis=0))
        featurePair = np.array(featurePair)

        model = load_model(modelDir)
        predY = model.predict_classes(featurePair)

        return predY[0]

    except Exception, e:
        if (str(e).split(" ")[0] == "Unable"):
            # Unable to find a face, return 0
            return 0


if __name__ == '__main__':
    Y = FaceVerifacation(img1Dir, img2Dir)
    print Y

print 'Done!'
