import cv2  
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow 

def find_face(image):
    im = image.copy()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    cnt_faces = face_cascade.detectMultiScale(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))

    for (x, y, w, h) in cnt_faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 8)
    cv2.imwrite('results/detect_face0.png', im)

    im = image.copy()
    for (x, y, w, h) in cnt_faces:
        strideX = w // 10
        strideY = h // 10
        im = im[y - strideY:y + h + strideY, x - strideX:x + w + strideX]
    cv2.imwrite('results/detect_face.png', im)
    return im


def canny(image):
    im = image.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.Canny(im, 55, 110)
    cv2.imwrite('results/edges.png', im)
    return im


def find_contours(image, original_image):
    im = image.copy()
    ans = np.zeros(original_image.shape, np.uint8)
    contours, _ = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, c in enumerate(contours):
        _, _, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10:
            cv2.drawContours(ans, contours, i, (255, 255, 255), 1)
    cv2.imwrite('results/contours.png', ans)
    return ans


def dilate(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im = cv2.equalizeHist(im)
    _, im = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    im = cv2.dilate(im, kernel)
    cv2.imwrite('results/dilate.png', im)
    return im


def gauss(image):
    im = cv2.GaussianBlur(image, (5, 5), 7) 
    normalized = np.zeros(im.shape, np.float32)
    normalized = cv2.normalize(im,  normalized, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    cv2.imwrite('results/gauss.png', im)
    return normalized


def bilaterial(image):
    im = image.copy()
    im = cv2.bilateralFilter(im, 15, 75, 75)
    cv2.imwrite('results/bilaterial.png', im)
    return im


def clarity(image):
    im = image.copy()
    gauss_img = cv2.GaussianBlur(im, (5, 5), 2) 
    im = cv2.addWeighted(im, 3.5, gauss_img, -2.5, 0)
    cv2.imwrite('results/inc_clarity.png', im)
    return im


def final_filter(original, image, image1, image2):
    im = image.copy()
    ans = np.zeros(original.shape, np.float32)
    f1 = image1.copy()
    f2 = image2.copy()

    for x in range(im.shape[0]):
        for y in range(im.shape[1]):
            for z in range(3):
                ans[x,y,z] = im[x, y] * f2[x, y, z] + (1 - im[x, y]) * f1[x, y, z]
                if ans[x,y,z] > 255:
                    ans[x,y,z] = 255
                elif ans[x,y,z] < 0:
                    ans[x,y,z] = 0
    cv2.imwrite('results/final_image.png', ans)
    return ans

def main():
    input_image = cv2.imread('lena.jpg')
    image = find_face(input_image)
    edges = canny(image)
    contours = find_contours(edges, image)
    dilate_img = dilate(contours)
    gauss_img = gauss(dilate_img)
    bil = bilaterial(image)
    clar = clarity(image)
    final_filter(image, gauss_img, bil, clar)

if __name__ == "__main__":
    main()