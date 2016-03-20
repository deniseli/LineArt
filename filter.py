# Make line drawings

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy import misc
from scipy import stats
from scipy import ndimage as ndi

from skimage import color
from skimage import feature

N = 7
sparse = "D"

class Line:
    def __init__(self, p, m):
        ''' point slope form '''
        self.m = m
        self.b = p.y - m * p.x
    def tostring(self):
        return str(m) + " " + str(b)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def tostring(im_seg):
    str = ""
    for val in im_seg.flatten():
        str += "1 " if val else "0 "
    return str[:-1]

def fromstring(str):
    g = lambda x: True if x == "1" else False
    vg = np.vectorize(g)
    return vg(np.array(str.split(" "))).reshape((N,N))

slope_cache = {}
def im_seg_to_slope(im_seg):
    im_str = tostring(im_seg)
    if im_str in slope_cache: return slope_cache[im_str]
    if not im_seg[N/2,N/2]: return None
    if not is_spanning_seg(im_seg):
        slope_cache[im_str] = None
        return None
    slope_cache[im_str] = calc_slope(im_seg)
    return slope_cache[im_str]

def is_spanning_seg(im_seg):
    return is_hspanning_seg(im_seg) or is_hspanning_seg(im_seg.T)

def is_hspanning_seg(im_seg):
    return np.sum(im_seg[:,:N/2]) > 0 and np.sum(im_seg[:,N/2+1:]) > 0

def calc_slope(im_seg):
    x = []
    y = []
    for i in range(N):
        for j in range(N):
            if im_seg[i,j]:
                x.append(i)
                y.append(j)
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.array(x),np.array(y))
    # possibly check r for noise
    return slope

def add_line(im, line, p):
    xmin = max(0, p.x - N**2)
    ymin = max(0, p.y - N**2)
    xmax = min(im.shape[0], p.x + N**2)
    ymax = min(im.shape[1], p.y + N**2)
    for x in range(xmin,xmax):
        y = line.m * x + line.b
        if y >= ymin and y < ymax:
            im[x,y] = 1

def filter(im, sparse):
    newim = np.zeros(im.shape)
    for i in range(im.shape[0]-N):
        for j in range(im.shape[1]-N):
            if sparse=="S" and (i%N > 0 or j%N > 0): continue # make sparse
            im_seg = im[i:i+N,j:j+N]
            m = im_seg_to_slope(im_seg)
            if m:
                p = Point(i+(N/2), j+(N/2))
                line = Line(p, m)
                add_line(newim, line, p)
    return newim

def makeoverlay(old, new):
    return old.astype(int) + new

if __name__ == "__main__":
    for S in range(1,6):
        im = color.rgb2gray(misc.imread('face.png'))
        edges = feature.canny(im, sigma=S)
        start = time.time()
        out = filter(edges, sparse)
        print "time elapsed: " + str(time.time() - start)
        plt.imshow(out, cmap=plt.cm.gray)
        plt.show()
        misc.imsave("face_N" + str(N) + "S" + str(S) + sparse + ".png", out)
