# hough_line_transform.py
'''
    Transform the image to hough space.
'''
import numpy as np
from collections import defaultdict
import pdb


def init_hough_img():
    return [0, []]


def hough_line_transform(img_bin, theta_res=1, rho_res=1, thr=100):
    '''
    input:
                img_bin: The edge image of the original image.
                theta_res: resolution of theta.
                rho_res: resolution of rho
                thr: threshold of the support of the (theta, rho) combination
    --------------------------------------------------------------------------
    output:
                hough_result:{(theta, rho):[support, [(y, x)...]]}
    '''
    print "hough start:\n"
    height, width = img_bin.shape
    theta_range = np.linspace(0.0, np.pi, np.ceil(180.0 / theta_res))
    rho_max = np.sqrt((width - 1) ** 2 + (height - 1)**2)
    rho_max = np.ceil(rho_max / rho_res)
    rho = np.linspace(-rho_max, rho_max, 2 * rho_max + 1)
    Hough_image = defaultdict(init_hough_img)
    for x in xrange(width - 1):
        for y in xrange(height - 1):
            if img_bin[y][x] > 0:
                for theta_id in xrange(len(theta_range)):
                    theta = theta_range[theta_id]
                    rhoVal = x * np.cos(theta) + y * np.sin(theta)
                    rhoIdx = np.argmin(np.abs(rho-rhoVal))
                    Hough_image[(theta_id, rhoIdx)][0] += 1
                    Hough_image[(theta_id, rhoIdx)][1].append((y, x))
                    # if Hough_image[(theta_id, rhoIdx)][0] > 100:
                    #     print "get "

    # Hough_image = sorted(Hough_image.items(), key=lambda (k, a): a[0])
    return Hough_image