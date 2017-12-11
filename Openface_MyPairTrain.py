#!/usr/bin/env python2
#
# Train a NN model to recognize two faces.
# By Xia
# 2017/08/22


import itertools
import os
import csv
from collections import Counter
import random
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.cross_validation import train_test_split

import numpy as np
np.set_printoptions(precision=3)


csvTrainDir = '/Users/xyq/PycharmProjects/faces/CSVFiles/openfaceOut128_07_500_367724_495201.csv'
csvTestDir = '/Users/xyq/PycharmProjects/faces/CSVFiles/openfaceOut128_06_500_272401_367005.csv'
csvPairsTrainDir = '/Users/xyq/PycharmProjects/faces/CSVFiles/openfaceOut128_07_pairs.csv'
csvPairsTestDir = '/Users/xyq/PycharmProjects/faces/CSVFiles/openfaceOut128_06_pairs.csv'
trainModelDir = '/Users/xyq/PycharmProjects/faces/Models_Openface/openface_train_model_test.h5'



def getFeaLabels(csvDir):
    features = []
    personIDs = []
    with open(csvDir, 'rb') as csvfile:
        # fieldnames = ['imgPath', '128features', 'personID']
        reader = csv.DictReader(csvfile)

        for row in reader:
            personIDs.append(int(row['personID']))
            feature_one_person = []
            fea = row['128features'].split('[')[1]
            fea = fea.split(']')[0]
            for one_feature in fea.split():
                feature_one_person.append(np.float(one_feature))
            features.append(feature_one_person)

    personIDs = np.array(personIDs)
    features = np.array(features)
    print 'Feature Matrix shape: {}'.format(features.shape)

    return features, personIDs


def makePairs(features, personIDs, csvPairsDir):
    nb_samples = len(personIDs)
    print 'nb_samples: ', nb_samples

    if not os.path.exists(csvPairsDir):
        print 'Start getting pairs...'
        with open(csvPairsDir, 'wb') as csvfile:
            fieldnames = ['pairFeatures', 'label']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            pair_data = []
            pair_labels = []
            pair_count = 1
            for i in range(nb_samples):
                for j in range(i+1, nb_samples):
                    pair_count += 1
                    if(personIDs[i] == personIDs[j]):
                        pair_data.append(np.append(features[i], features[j], axis=0))
                        pair_labels.append(1)
                        writer.writerow({'pairFeatures': np.append(features[i], features[j], axis=0), 'label': 1})
                    elif(pair_count%250 == 0):
                        pair_data.append(np.append(features[i], features[j], axis=0))
                        pair_labels.append(0)
                        writer.writerow({'pairFeatures': np.append(features[i], features[j], axis=0), 'label': 0})

            pair_data = np.array(pair_data)
            pair_labels = np.array(pair_labels)

        print 'Got the pairs!'
        print 'Pair features shape: {}'.format(pair_data.shape)
        count = Counter(pair_labels)
        print(count)
        print 'Saved the pairs.'
        return pair_data, pair_labels

    else:
        print 'Pairs csv file already exists.'
        print 'Start extracting from csv file...'
        pair_data = []
        pair_labels = []
        with open(csvPairsDir, 'rb') as csvfile:
            # fieldnames = ['imgPath', '128features', 'personID']
            reader = csv.DictReader(csvfile)

            for row in reader:
                pair_labels.append(int(row['label']))
                feature_one_pair = []
                fea = row['pairFeatures'].split('[')[1]
                fea = fea.split(']')[0]
                for one_feature in fea.split():
                    feature_one_pair.append(np.float(one_feature))
                pair_data.append(feature_one_pair)
        print 'Got the train pairs!'
        pair_labels = np.array(pair_labels)
        pair_data = np.array(pair_data)
        print 'Feature Matrix shape: {}'.format(pair_data.shape)
        count = Counter(pair_labels)
        print(count)
        return pair_data, pair_labels


def modelTrain(pair_data, pair_labels, trainModelDir):
    if not os.path.exists(trainModelDir):
        print 'Starting training the model...'

        train_pair, validation_pair, train_pair_labels, validation_pair_labels = train_test_split(pair_data, pair_labels,
                                                                                              test_size=0.2,
                                                                                              random_state=random.randint(0, 100))
        # Train
        # Create a model
        model = Sequential()
        # Input
        model.add(Dense(64, activation='relu', input_dim=256))
        model.add(Dropout(0.5))
        # Hidden
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        # Output
        model.add(Dense(2, activation='softmax'))

        # SGD
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.fit(train_pair, train_pair_labels,
                  epochs=10, batch_size=200,
                  validation_data=(validation_pair, validation_pair_labels))

        print('Saving model ...')
        model.save(trainModelDir)
        print('Model Saved.')
    else:
        print 'Model already exsits.'


def modelEvaluate(pair_data_test, pair_labels_test, trainModelDir):
    model = load_model(trainModelDir)
    print 'Loaded the model.'
    print 'Starting evaluating...'

    score = model.evaluate(pair_data_test, pair_labels_test, batch_size=100)
    print model.metrics_names
    print score



if __name__ == '__main__':
    features_train, personIDs_train = getFeaLabels(csvTrainDir)
    pair_data_train, pair_labels_train = makePairs(features_train, personIDs_train, csvPairsTrainDir)
    modelTrain(pair_data_train, pair_labels_train, trainModelDir)

    features_test, personIDs_test = getFeaLabels(csvTestDir)
    pair_data_test, pair_labels_test = makePairs(features_test, personIDs_test, csvPairsTestDir)
    modelEvaluate(pair_data_test, pair_labels_test, trainModelDir)



print 'Done!'
