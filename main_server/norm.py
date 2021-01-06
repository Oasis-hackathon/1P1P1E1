import numpy as np

# Normalization
def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0)) * (x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

# Stack
def stack(X):
    X0 = X[:,:,0]
    X1 = X[:,:,1]
    X2 = X[:,:,2]

    X_scale_0 = scale(X0, -1, 1)
    X_scale_1 = scale(X1, -1, 1)
    X_scale_2 = scale(X2, -1, 1)

    return np.stack((X_scale_0, X_scale_1, X_scale_2), axis=2)

# filter
sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0