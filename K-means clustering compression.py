import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#Read img
img = cv.imread('wibu.jpg',1)
cv.imshow('orig', img)
z = img.reshape((-1,3))
z = np.float32(z)

#Calculate KMeans
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
flags = cv.KMEANS_PP_CENTERS
compactness, labels, centers = cv.kmeans(z,5, None, criteria, 10, flags)

#Label clusters
centers = np.uint8(centers)
img2 = centers[labels]
img2 = img2.reshape((img.shape))

cv.imshow('kmeans', img2)
cv.waitKey(0)
cv.destroyAllWindows()
