from PIL import Image, ImageDraw
from collections import defaultdict
from math import floor, ceil

import numpy as np
import ctypes
import time
from ctypes import *
from numpy.ctypeslib import ndpointer

def get_cuda_tetration(sharpness):
    dll = ctypes.CDLL('./powtowfrac.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.create_frame
    func.argtypes = [c_int, c_double, c_double, c_double, c_int, POINTER(c_double)]
    func.restype = ndpointer(dtype=ctypes.c_double, shape=(sharpness * sharpness,))
    return func

def cuda_tetration(sharpness, centerRe, centerIm, epsilon, maxIter, res, N, tetr):
    res_p = res.ctypes.data_as(POINTER(c_double))

    return tetr(sharpness, centerRe, centerIm, epsilon, maxIter, res_p)

def linear_interpolation(color1, color2, t):
    return color1 * (1 - t) + color2 * t 

def tetr_execute(sRe, sIm, sEpsilon, sMaxiter):
    start = time.time()

    # Input variables
    sharpness = 750

    re = float(sRe)
    im = float(sIm)

    epsilon = float(sEpsilon)

    maxIter = int(sMaxiter)

    # Calculation of window size, number of pixels
    reStart = re - epsilon
    reEnd = re + epsilon
    imStart = im - epsilon
    imEnd = im + epsilon

    pngWidth = int(sharpness)
    pngHeight = int(pngWidth * (imEnd - imStart) / (reEnd - reStart))

    N = pngWidth * pngHeight

    # Execution
    __cuda_tetration = get_cuda_tetration(sharpness)
    res = np.zeros(N).astype('float64')
    res = cuda_tetration(sharpness, re, im, epsilon, maxIter, res, N, __cuda_tetration)
    elapsed = time.time() - start
    start = time.time()
    print "Time elapsed:", elapsed

    # Put results in picture
    print("Start creating image output.png")
    pic = Image.new('HSV', (pngWidth, pngHeight), (0, 0, 0))
    draw = ImageDraw.Draw(pic)

    black = (0,0,0)
    palette = range(256)
    for i in palette:
        h = i
        s = 255
        v = 255 if i < 255 else 0
        palette[i] = (h,s,v)

    histogram = defaultdict(lambda: 0)

    it = res

    # Create histogram of colors
    for T in xrange(N):
        if it[T] < maxIter:
            histogram[int(it[T])] += 1

    total = sum(histogram.values())
    hues = []
    h = 0
    for i in range(maxIter):
        h += (255 * histogram[i])
        hues.append(h)
    hues.append(h)

    if total > 0:
        hues = [hue / total for hue in hues]

    # Draw to image
    for T in xrange(N):
        x = T % pngHeight
        y = T / pngHeight
        shade = int(linear_interpolation(hues[int(floor(it[T]))], hues[int(ceil(it[T]))], it[T] % 1))
        color = palette[shade] if it[T] < maxIter else black
        draw.point([x,y], color)

    pic.convert('RGB').save('output.png', 'PNG')

    elapsed = time.time() - start
    print "Time elapsed:", elapsed