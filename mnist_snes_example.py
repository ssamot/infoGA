

from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from keras_helper import NNWeightHelper
from snes import SNES





# use just a small sample of the train set to test
SAMPLE_SIZE = 1024
# how many different sets of weights ask() should return for evaluation
POPULATION_SIZE = 10
# how many times we will loop over ask()/tell()
GENERATIONS = 300

def train_classifier(model, X, y):
    X_features = model.predict(X)
    #clf = ExtraTreesClassifier(n_estimators=100, n_jobs=4)
    clf = DecisionTreeClassifier()

    clf.fit(X_features, y)
    y_pred = clf.predict(X_features)
    return clf, y_pred


def predict_classifier(model, clf, X):
    X_features = model.predict(X)
    return clf.predict(X_features)

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

# this is irrelevant for what we want to achieve
model.compile(loss="mse", optimizer="adam")
print("compilation is over")

nnw = NNWeightHelper(model)
weights = nnw.get_weights()


def main():
    print("Total number of weights to evolve is:", weights.shape)

    all_examples_indices = list(range(x_train.shape[0]))

    clf, _ = train_classifier(model, x_train, y_train)

    y_pred = predict_classifier(model, clf, x_test)
    print(y_test.shape, y_pred.shape)
    test_accuracy = accuracy_score(y_test, y_pred)

    print('Non-trained NN Test accuracy:', test_accuracy)
    # print('Test MSE:', test_mse)

    snes = SNES(weights, 1, POPULATION_SIZE)
    for i in range(0, GENERATIONS):
        start = timer()
        asked = snes.ask()

        # to be provided back to snes
        told = []
        # use a small number of training samples for speed purposes
        subsample_indices = np.random.choice(all_examples_indices, size=SAMPLE_SIZE, replace=False)
        # evaluate on another subset
        subsample_indices_valid = np.random.choice(all_examples_indices, size=SAMPLE_SIZE + 1, replace=False)

        # iterate over the population
        for asked_j in asked:
            # set nn weights
            nnw.set_weights(asked_j)
            # train the classifer and get back the predictions on the training data
            clf, _ = train_classifier(model, x_train[subsample_indices], y_train[subsample_indices])

            # calculate the predictions on a different set
            y_pred = predict_classifier(model, clf, x_train[subsample_indices_valid])
            score = accuracy_score(y_train[subsample_indices_valid], y_pred)

            # clf, _ = train_classifier(model, x_train, y_train)
            # y_pred = predict_classifier(model, clf, x_test)
            # score = accuracy_score(y_test, y_pred)
            # append to array of values that are to be returned
            told.append(score)

        snes.tell(asked, told)
        end = timer()
        print("It took", end - start, "seconds to complete generation", i + 1)

    nnw.set_weights(snes.center)

    clf, _ = train_classifier(model, x_train, y_train)
    y_pred = predict_classifier(model, clf, x_test)

    print(y_test.shape, y_pred.shape)
    test_accuracy = accuracy_score(y_test, y_pred)

    print('Test accuracy:', test_accuracy)


if __name__ == '__main__':
    main()
