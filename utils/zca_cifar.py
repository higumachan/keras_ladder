from keras.datasets import cifar10
import numpy as np

filepath = '../data/'

def whitening_zca(data):
    n_components, n_channels, height, width = data.shape
    data = np.reshape(data, (n_components, n_channels*height*width))
    X = data - np.mean(data, axis=0)
    C = np.dot(X.T, X)/float(n_components)
    U,S,V = np.linalg.svd(C)
    epsilon = 0.1
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(S + epsilon))), U.T)
    fixed_data = np.dot(X, ZCAMatrix.T)
    fixed_data = np.reshape(fixed_data, (n_components, n_channels, height, width))
    return fixed_data

def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    print "X_train:", type(X_train), X_train.shape
    X_train = whitening_zca(X_train)
    X_test = whitening_zca(X_test)

    np.save(filepath+'X_train.npy', X_train)
    np.save(filepath+'y_train.npy', y_train)
    np.save(filepath+'X_test.npy', X_test)
    np.save(filepath+'y_test.npy', y_test)


if __name__ == '__main__':
    main()
