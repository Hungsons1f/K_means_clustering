import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


#Create a random array of 50 points
x = np.random.randint(25,100,25)
y = np.random.randint(175,255,25)
z = np.hstack((x,y))
z = np.resize(z,(50,1))
z = np.float32(z)

#Calculate KMeans
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
flags = cv.KMEANS_PP_CENTERS
compactness, labels, centers = cv.kmeans((z,c),2, None, criteria, 10, flags)

#Label clusters
A = z[labels == 0]
B = z[labels == 1]

#Plot data
plt.hist(z, bins=256, range=[0,256])
plt.hist(A, bins=256, range=[0,256], color = 'b')
plt.hist(B, bins=256, range=[0,256], color = 'r')
plt.hist(centers, bins=32, range=[0,256], color = 'g')
plt.show()
