# Canny_Hough_line.py
'''
This program is to detect the line segment from an image.
Canny edge detection algorithm is used to detect the edge.
Hough line detection algorithm is used to detect the line.
In order to decrease the time of computation, both method is
combined.
'''
import cv2
import numpy as np
import HoughTransform


def traverse(i, j, img_h, img_l):
    '''
    Traverse the 8-neighbor of the (i, j) point of high magnitute image
    of sobel result. If there is any low sobel point in its neighbor, it is
    selected and this traverse is passed to the new sobel point.
    '''
    neighbor = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in neighbor:
        if img_h[i + dy][j + dx] == 0 and img_l[i + dy][j + dx]:
            img_h[i + dy][j + dx] = 250
            traverse(i + dy, j + dx, img_h, img_l)

img = cv2.imread('./Image/Corridor_l_low.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# sobel
grad_x = np.zeros(gray.shape, dtype=float)
grad_y = np.zeros(gray.shape, dtype=float)
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]
sobel_x = np.array(sobel_x)
sobel_x = sobel_x.reshape(9)
sobel_y = [[-1, -2, -1],
           [0, 0, 0],
           [1, 2, 1]]
sobel_y = np.array(sobel_y)
sobel_y = sobel_y.reshape(9)
width = gray.shape[1]
height = gray.shape[0]

for x in xrange(1, width - 1):
    for y in xrange(1, height - 1):
        neigbor = gray[y - 1:y + 2, x - 1:x + 2]
        d_x = np.dot(sobel_x, neigbor.reshape(9))
        d_y = np.dot(sobel_y, neigbor.reshape(9))
        grad_x[y][x] = d_x
        grad_y[y][x] = d_y
sobel_mag = np.hypot(grad_x, grad_y)
sobel_dir = np.arctan2(grad_y, grad_x)
mag_sup = sobel_mag

dirs = {('v', 0): [67.5 * np.pi / 180, 112.5 * np.pi / 180],
        # ('h', 3): [[0, 22.5 * np.pi / 180], [157.5*np.pi / 180, np.pi]],
        ('up_r', 1): [22.5 * np.pi / 180, 67.5 * np.pi / 180],
        ('up_l', 2): [112.5 * np.pi / 180, 157.5 * np.pi / 180]}
# round the diretions to one of four directions
for x in xrange(1, width - 1):
    for y in xrange(1, height - 1):
        for key, agree in dirs.iteritems():
            if agree[0] <= sobel_dir[y][x] < agree[1] or\
                    -agree[1] <= sobel_dir[y][x] < -agree[0]:
                sobel_dir[y][x] = key[1]
                break
        else:
            sobel_dir[y][x] = 3
# Non Maximum Suppression
for x in xrange(1, width - 1):
    for y in xrange(1, height - 1):
        if sobel_dir[y][x] == 0:  # vertical
            if sobel_mag[y][x] <= sobel_mag[y][x + 1] or \
                    sobel_mag[y][x] <= sobel_mag[y][x - 1]:
                mag_sup[y][x] = 0
        elif sobel_dir[y][x] == 1:  # up right
            if sobel_mag[y][x] <= sobel_mag[y - 1][x + 1] or \
                    sobel_mag[y][x] <= sobel_mag[y + 1][x - 1]:
                mag_sup[y][x] = 0
        elif sobel_dir[y][x] == 2:  # up left
            if sobel_mag[y][x] <= sobel_mag[y - 1][x - 1] or\
                    sobel_mag[y][x] <= sobel_mag[y + 1][x + 1]:
                mag_sup[y][x] = 0
        else:  # horizon
            if sobel_mag[y][x] <= sobel_mag[y][x + 1] or\
                    sobel_mag[y][x] <= sobel_mag[y][x - 1]:
                mag_sup[y][x] = 0

# Edge Linking
maximum = np.max(mag_sup)
th = 0.2 * maximum
tl = 0.1 * maximum
img_high = np.zeros(gray.shape)
img_low = np.zeros(gray.shape)
for x in xrange(width - 1):
    for y in xrange(height - 1):
        if mag_sup[y][x] >= tl:
            if mag_sup[y][x] >= th:
                img_high[y][x] = mag_sup[y][x]
            else:
                img_low[y][x] = mag_sup[y][x]

for x in xrange(1, width - 1):
    for y in range(1, height - 1):
        if img_high[y][x] > 0:
            traverse(y, x, img_high, img_low)

Hough_image = HoughTransform.hough_line_transform(img_high)
hough = sorted(Hough_image.items(), key=lambda(k, a): a[0], reverse=True)

red = 0
blue = 0
img_1 = np.copy(img)
for p in hough:
    if p[1][0] > 200 and p[0][0] > 170 and red < 2:
        print "get red"
        red += 1
        for y, x in p[1][1]:
            img_1[y][x][0] = 0
            img_1[y][x][1] = 0
            img_1[y][x][2] = 255
    elif p[1][0] > 150 and p[0][0] < 30 and blue < 8:
        print "get blue"
        blue += 1
        for y, x in p[1][1]:
            img_1[y][x][0] = 255
            img_1[y][x][1] = 0
            img_1[y][x][2] = 0
    if red >= 2 and blue >= 8:
        break

for p in hough:
    if p[1][0] > 150 and 10<p[0][0] <35:
        print "get green"
        for y, x in p[1][1]:
            img_1[y][x][0] = 0
            img_1[y][x][1] = 255
            img_1[y][x][2] = 0

for p in hough:
    if p[1][0] > 200 and 125 < p[0][0] < 145:
        print "get orange"
        for y, x in p[1][1]:
            img_1[y][x][0] = 0
            img_1[y][x][1] = 255
            img_1[y][x][2] = 255

cv2.imwrite('./tmp_l.jpg', img_1)

    # cv2.imshow('gray', gray)
