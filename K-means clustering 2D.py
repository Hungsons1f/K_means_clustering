import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#Create a random array of 50 points
x = np.random.randint(25,125,(25,2))
y = np.random.randint(130,255,(25,2))
z = np.vstack((x,y))
z = np.float32(z)

#Calculate KMeans
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
flags = cv.KMEANS_PP_CENTERS
compactness, labels, centers = cv.kmeans(z,2, None, criteria, 10, flags)

#Label clusters
A = z[labels.ravel() == 0]
B = z[labels.ravel() == 1]

#Plot data
plt.scatter(A[:,0],A[:,1], c='b', marker="o")
plt.scatter(B[:,0],B[:,1], c='r', marker="^")
plt.scatter(centers[:,0],centers[:,1], c='g', marker="P")
plt.show()
