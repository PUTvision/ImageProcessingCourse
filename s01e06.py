import sklearn
from sklearn import tree, datasets, cluster
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def ex_0():
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [0, 0, 0, 1]
    clf = sklearn.tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print(clf.predict([[1, 1]]))

    dict = {"VW": 0, "Ford": 1, "Opel": 2}
    print(dict["VW"])


def ex_1():
    iris = sklearn.datasets.load_iris()
    print(iris.data)
    print(iris.target)

    X = iris.data
    k_means = sklearn.cluster.KMeans(n_clusters=3)
    k_means.fit(X)

    labels = k_means.labels_
    fig = plt.figure()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    plt.show()


def ex_2():
    digits = sklearn.datasets.load_digits()

    images_and_labels = list(zip(digits.images, digits.target))
    for index, (image, label) in enumerate(images_and_labels[:4]):
        plt.subplot(2, 4, index + 1)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %i' % label)

    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    classifier = sklearn.svm.LinearSVC()
    classifier.fit(data[:np.int(n_samples / 2)], digits.target[:np.int(n_samples / 2)])
    expected = digits.target[np.int(n_samples / 2):]
    predicted = classifier.predict(data[np.int(n_samples / 2):])

    print("Classification report for classifier %s:\n%s\n"
          % (classifier, sklearn.metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % sklearn.metrics.confusion_matrix(expected, predicted))

    images_and_predictions = list(zip(digits.images[int(n_samples / 2):], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

    plt.show()


if __name__ == "__main__":
    ex_0()
    #ex_1()
    #ex_2()
