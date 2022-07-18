import cv2
import numpy as np

np.set_printoptions(linewidth=np.inf, precision=2, threshold=np.inf)

image = cv2.imread('proj.jpg')
r, corners = cv2.findChessboardCorners(image, (8, 6))
cv2.drawChessboardCorners(image, (8, 6), corners, r)

points_3d = np.empty((0, 4), float)
for i in range(8):
    for j in range(6):
        point = np.array([[0.1 * i + 1, 0.1 * j + 1, 1.0, 1]])
        points_3d = np.append(points_3d, point, axis=0)

# cv2.imshow('',image)
# cv2.waitKey(0)

a = np.empty((0, 12))
for i in range(len(corners)):
    u = corners[i][0][0]
    v = corners[i][0][1]
    x = points_3d[i][0]
    y = points_3d[i][1]
    z = points_3d[i][2]

    strip = np.array(
        [[x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u],
         [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v]])

    a = np.append(a, strip, axis=0)

eigenvalues, eigenvectors = np.linalg.eig(np.dot(a.T, a))
projection = eigenvectors[:, np.argmin(eigenvalues)]
projection = np.reshape(projection, (3, 4))

print(np.dot(projection, points_3d[0]))
print(corners[0])
