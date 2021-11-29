import cv2  
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave

def ReLu(a):
    return np.maximum(0, a)

def SoftMax(x):
  return np.exp(x) / sum(np.exp(x))

def MaxPooling(image, pool_sz):
    W = image.shape[1] // pool_sz
    H = image.shape[2] // pool_sz
    _1 = image.shape[0]

    ans = np.empty((_1, W, H), dtype=np.float32)

    for m in range(_1):
        for x in range(W):
            for y in range(H):
                gg = -1
                for xx in range(x * pool_sz, (x + 1) * pool_sz):
                    for yy in range(y * pool_sz, (y + 1) * pool_sz):
                        gg = np.maximum(gg, image[m, xx, yy])
                ans[m, x, y] = gg
    return ans

def BatchNormalization(X, gamma, beta, epsilon):
    mean = X.mean(axis=0)
    mean2 = ((X - mean)**2).mean(axis=0)
    
    X_ = (X - mean) / np.sqrt(mean2 + epsilon)
    return gamma * X_ + beta

def make_convolution(image, filter_cnt, filter_R, filter_S, filter_C):
    W = image.shape[0]
    H = image.shape[1]
    Z = image.shape[2]

    filters = np.random.uniform(size=(filter_cnt, filter_C, filter_R, filter_S))
    ans = np.zeros((filter_cnt, W, H), dtype=np.float32)

    for fil in range(filter_cnt):
        for x in range(W):
            for y in range(H):
                for i in range(filter_R):
                    for j in range(filter_S):
                        for k in range(filter_C):
                            xx = min(x + i, H - 1)
                            yy = min(y + j, Z - 1)
                            ans[fil][x][y] += image[k, xx, yy] * filters[fil, k, i, j]
    return ans

def make_normalization(data, filter_cnt):
    for i in range(filter_cnt):
        data[i] = BatchNormalization(data[i], 1, 0, 1e-9)
    return data

def make_relu(data, filter_cnt, W, H):
    for i in range(filter_cnt):
        for x in range(W):
            for y in range(H):
                data[i, x, y] = ReLu(data[i, x, y])
    return data

def make_softmax(data, filter_cnt):
    ans = data
    for i in range(filter_cnt):
        ans = SoftMax(data[i])
    return ans

def main():
    input_image = cv2.imread('dog.jpg')
    print(input_image.shape)
    convolution = make_convolution(input_image, 5, 3, 3, 3)
    norm_convolution = make_normalization(convolution, 5)
    relu_convolution = make_relu(norm_convolution, 5, input_image.shape[0], input_image.shape[1])
    ans = MaxPooling(relu_convolution, 2)

    softmax = make_softmax(ans, 5)

    imsave('results/0.jpg', ans[0])
    imsave('results/1.jpg', ans[1])
    imsave('results/2.jpg', ans[2])
    imsave('results/3.jpg', ans[3])
    imsave('results/4.jpg', ans[4])

    print(softmax[0])


if __name__ == "__main__":
    main()