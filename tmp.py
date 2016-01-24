def traverse(i, j, img_h, img_l):
    '''
    Traverse the 8-neighbor of the (i, j) point of high magnitute image
    of sobel result. If there is any low sobel point in its neighbor, it is
    selected and this traverse is passed to the new sobel point.
    '''
    neighbor = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                (0, 1), (1, -1), (1, 0), (1, 1)]
    for dy, dx in neighbor:
        if img_h[i + dy][j + dx] == 0 and img_l[i+dy][j + dx]:
            img_h[i + dy][j + dx] = 250
            traverse(i+dy, j+dx, img_h, img_l)

for x in xrange(1, width-1):
    for y in range(1, height-1):
        if img_high[y][x] > 0:
            traverse(y, x, img_high, img_low)
cv2.imwrite('./tmp2.jpg', img_high)