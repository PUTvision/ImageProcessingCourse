from typing import Tuple

import os
import os.path as osp
import pickle


import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from skimage import data

SEED = 42
NB_WORDS = 20
DESIRED_WIDTH = 1024


# data loading
def load_dataset(dataset_dir_path: str) -> Tuple[np.array, np.array]:
    X, y = [], []
    for i, class_dir in enumerate(os.listdir(dataset_dir_path)):
        class_dir_path = osp.join(dataset_dir_path, class_dir)
        for file in os.listdir(class_dir_path):
            img_file = cv2.imread(osp.join(class_dir_path, file), cv2.IMREAD_GRAYSCALE)
            if img_file is None:
                print(osp.join(class_dir_path, file))
            # cv2.imshow('original', img_file)
            height, width = img_file.shape
            scale_factor = DESIRED_WIDTH / width
            img_rescaled = cv2.resize(img_file, (DESIRED_WIDTH, int(height*scale_factor)))
            # cv2.imshow('rescaled', img_rescaled)
            # cv2.waitKey(0)

            X.append(img_rescaled)
            y.append(i)

    X = np.array(X)
    y = np.array(y)
    return X, y


def prepare_vocabulary(images, feature_detector_descriptor, nb_words):
    features = []
    for image in images:
        print(image.shape)
        keypoints, image_descriptor = feature_detector_descriptor.detectAndCompute(image, None)
        print(len(keypoints))
        features.extend(image_descriptor)
    features = np.float32(features)
    kmeans = KMeans(n_clusters=nb_words, random_state=42, n_jobs=-1).fit(features)
    return kmeans


def descriptor2histogram(descriptor, vocab_model, normalize=True):
    features_words = vocab_model.predict(descriptor)
    histogram = np.zeros(vocab_model.n_clusters, dtype=np.float32)
    unique, counts = np.unique(features_words, return_counts=True)
    histogram[unique] += counts
    if normalize:
        histogram /= histogram.sum()
    return histogram


def apply_feature_transform(X, feature_detector_descriptor, vocab_model):
    X_transformed = []
    for image in X:
        keypoints, image_descriptor = feature_detector_descriptor.detectAndCompute(image, None)
        bow_features_histogram = descriptor2histogram(image_descriptor, vocab_model)
        X_transformed.append(bow_features_histogram)
    X_transformed = np.array(X_transformed)
    return X_transformed


def print_score(clf, X_train, y_train, X_test, y_test):
    # val: {clf.score(X_val, y_val)},
    print(
        f'train: {clf.score(X_train, y_train)}, test: {clf.score(X_test, y_test)}'
    )


def lab():
    feature_detector_descriptor = cv2.AKAZE_create()

    # image = data.astronaut()[:, :, ::-1]
    #
    # keypoints, image_descriptor = feature_detector_descriptor.detectAndCompute(image, None)
    # # Drugim argumentem może być maska binarna, która służy do zawężenia obszaru z którego
    # # uzyskujemy punkty kluczowe/deskryptor – jako, że nam akurat zależy na całym obrazie,
    # # podaliśmy tam wartość None.
    #
    # # Wynikiem jest lista punktów kluczowych oraz odpowiadającym im wektorów – które składają się na deskryptor.
    # print(f'{len(keypoints)}, {len(image_descriptor)}')
    # print(f'{image_descriptor[0]}')
    #
    # X, y = load_dataset(DATA_DIR_PATH)

    X, y = load_dataset('./../_data/zpo/s01e03/dataset_1')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f'len(X_train): {len(X_train)}, len(X_test): {len(X_test)}')

    vocab_model = prepare_vocabulary(X_train, feature_detector_descriptor, NB_WORDS)

    X_train = apply_feature_transform(X_train, feature_detector_descriptor, vocab_model)
    X_test = apply_feature_transform(X_test, feature_detector_descriptor, vocab_model)

    svc = SVC(random_state=SEED)
    print(svc.fit(X_train, y_train))
    print_score(svc, X_train, y_train, X_test, y_test)

    mlp = MLPClassifier(random_state=SEED)
    print(mlp.fit(X_train, y_train))
    print_score(mlp, X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(random_state=SEED)
    print(rf.fit(X_train, y_train))
    print_score(rf, X_train, y_train, X_test, y_test)


def project():
    X, y = load_dataset('./../_data/zpo/s01e03/dataset_2_fixed/train')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    # )  # W drugim problemie testowy wrzuci się potem, X_test to w zasadzie walidacyjny

    feature_detector_descriptor = cv2.AKAZE_create()

    # vocab_model = prepare_vocabulary(X_train, feature_detector_descriptor, NB_WORDS)
    #
    # X_train = apply_feature_transform(X_train, feature_detector_descriptor, vocab_model)
    # # X_val = apply_feature_transform(X_val, feature_detector_descriptor, vocab_model)
    # X_test = apply_feature_transform(X_test, feature_detector_descriptor, vocab_model)
    #
    # pickle.dump(vocab_model, open(f'./../_data/zpo/s01e03/vocab_model_{NB_WORDS}.p', 'wb'))
    # pickle.dump(X_train, open(f'./../_data/zpo/s01e03/X_train_{NB_WORDS}.p', 'wb'))
    # # pickle.dump(X_val, open(f'./../_data/zpo/s01e03/X_val_{NB_WORDS}.p', 'wb'))
    # pickle.dump(X_test, open(f'./../_data/zpo/s01e03/X_test_{NB_WORDS}.p', 'wb'))

    vocab_model = pickle.load(open(f'./../_data/zpo/s01e03/vocab_model_{NB_WORDS}.p', 'rb'))
    X_train = pickle.load(open(f'./../_data/zpo/s01e03/X_train_{NB_WORDS}.p', 'rb'))
    # X_val = pickle.load(open(f'./../_data/zpo/s01e03/X_val_{NB_WORDS}.p', 'rb'))
    X_test = pickle.load(open(f'./../_data/zpo/s01e03/X_test_{NB_WORDS}.p', 'rb'))

    svc = SVC(random_state=SEED)
    print(svc.fit(X_train, y_train))
    print_score(svc, X_train, y_train, X_test, y_test)

    mlp = MLPClassifier(random_state=SEED)
    print(mlp.fit(X_train, y_train))
    print_score(mlp, X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(random_state=SEED)
    print(rf.fit(X_train, y_train))
    print_score(rf, X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    lab()
    # project()
