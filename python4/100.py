import numpy as np

A = np.array([[4, 9, 9], [9, 1, 6], [9, 2, 3]])
print("Matrix A (3 by 3):\n", A)

B = np.array([[2, 2], [5, 7], [4, 4]])
print("Matrix B (3 by 2):\n", B)

np.matmul(A, B)

A @ B

try:
    np.matmul(B, A)
except ValueError as err:
    print(err)


x = np.array([1, -2, -5])
y = np.array([4, 3, -1])

print("Shape of vector x:", x.shape)
print("Number of dimensions of vector x:", x.ndim)
print("Shape of vector x, reshaped to a matrix:", x.reshape((3, 1)).shape)
print("Number of dimensions of vector x, reshaped to a matrix:", x.reshape((3, 1)).ndim)

np.matmul(x,y)
np.dot(A, B)

A - 2

#***********************************************************************************************************************

import numpy as np
import cv2


def T(v):
    w = np.zeros((3, 1))
    w[0, 0] = 3 * v[0, 0]
    w[2, 0] = -2 * v[1, 0]

    return w


v = np.array([[3], [5]])
w = T(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

u = np.array([[1], [-2]])
v = np.array([[2], [4]])

k = 7

print("T(k*v):\n", T(k*v), "\n k*T(v):\n", k*T(v), "\n\n")
print("T(u+v):\n", T(u+v), "\n T(u)+T(v):\n", T(u)+T(v))


def L(v):
    A = np.array([[3, 0], [0, 0], [0, -2]])
    print("Transformation matrix:\n", A, "\n")
    w = A @ v

    return w

v = np.array([[3], [5]])
w = L(v)

print("Original vector:\n", v, "\n\n Result of the transformation:\n", w)

#***************************************************************************************************
def T_hscaling(v):
    A = np.array([[2, 0], [0, 1]])
    w = A @ v

    return w


def transform_vectors(T, v1, v2):
    V = np.hstack((v1.reshape(2, 1), v2.reshape(2, 1)))
    W = T(V)

    return W


e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])

transformation_result_hscaling = transform_vectors(T_hscaling, e1, e2)

print("Original vectors:\n e1= \n", e1, "\n e2=\n", e2,
      "\n\n Result of the transformation (matrix form):\n", transformation_result_hscaling)

#*****************************************************************************************************

img = cv2.imread('images/leaf_original.png', 0)
plt.imshow(img)

image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(image_rotated)

image_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

plt.imshow(image_rotated)

rows,cols = image_rotated.shape
M = np.float32([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
image_rotated_sheared = cv2.warpPerspective(image_rotated, M, (int(cols), int(rows)))
plt.imshow(image_rotated_sheared)

image_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)

image_sheared = cv2.warpPerspective(img, M, (int(cols), int(rows)))
image_sheared_rotated = cv2.rotate(image_sheared, cv2.ROTATE_90_CLOCKWISE)
plt.imshow(image_sheared_rotated)

M_rotation_90_clockwise = np.array([[0, 1], [-1, 0]])
M_shear_x = np.array([[1, 0.5], [0, 1]])

print("90 degrees clockwise rotation matrix:\n", M_rotation_90_clockwise)
print("Matrix for the shear along x-axis:\n", M_shear_x)

print("M_rotation_90_clockwise by M_shear_x:\n", M_rotation_90_clockwise @ M_shear_x)
print("M_shear_x by M_rotation_90_clockwise:\n", M_shear_x @ M_rotation_90_clockwise)