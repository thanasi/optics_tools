##########################################################
# WINDOWS ONLY
# tool to help align a beam expander as described here:
#
# http://www.uslasercorp.com/envoy/diverge.html
#
##########################################################
import platform
assert platform.system() == "Windows", "Currently only Windows is supported"

import matplotlib as mpl
mpl.use("wx")

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_isodata
from skimage import color

import scipy.ndimage as ndi
from scipy.ndimage import filters

import pymba
import time

W0 = 50


def init_camera(vimba):
    system = vimba.getSystem()

    if system.GeVTLIsPresent:
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)

    cameraIds = vimba.getCameraIds()

    camera0 = vimba.getCamera(cameraIds[0])
    camera0.openCamera()

    camera0.AcquisitionMode = 'SingleFrame'
    camera0.PixelFormat = 'Mono8'
    camera0.Width = camera0.WidthMax
    camera0.Height = camera0.HeightMax
    camera0.OffsetX = 0
    camera0.OffsetY = 0
    camera0.BlackLevel = 0
    camera0.Gain = 0
    camera0.ExposureTimeAbs = camera0.ExposureAutoMin
    # camera0.ExposureTimeAbs = 105

    frame0 = camera0.getFrame()
    frame0.announceFrame()
    camera0.startCapture()

    return system, camera0, frame0

def get_frame(camera, frame):

    H,W = frame.height, frame.width
    frame.queueFrameCapture()
    camera.runFeatureCommand('AcquisitionStart')
    camera.runFeatureCommand('AcquisitionStop')
    frame.waitFrameCapture()
    imgData = np.ndarray(buffer = frame.getBufferByteData(),
                         dtype = np.uint8,
                         shape = (H,W,1))[:,:,0]

    return imgData

def cleanup_cam(camera):
    camera.endCapture()
    camera.revokeAllFrames()
    camera.closeCamera()


def circle(x0,y0,r,num_points=1000):
    theta = np.arange(0,2*np.pi, 2*np.pi/num_points)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0

    return x,y

def analyze_beam(im):
    Y,X = np.mean(np.where(im==im.max()),1)
    sl = (slice(int(Y)-W0,int(Y)+W0+1), slice(int(X)-W0,int(X)+W0+1))

    im2 = ndi.gaussian_filter(im[sl],3)

    th = threshold_otsu(im2)
    mask = im2 > th

    # mask = ndi.binary_fill_holes(mask)

    Y1,X1 = ndi.center_of_mass(mask)
    Y = int(Y1) + sl[0].start
    X = int(X1) + sl[1].start

    edge_mask = np.logical_xor(mask ,ndi.binary_erosion(mask))
    y,x = np.mgrid[:2*W0+1, :2*W0+1]
    r = np.sqrt((x-X1)**2 + (y-Y1)**2) * edge_mask
    D = np.mean(r[r>0])

    return X,Y,D


if __name__ == "__main__":

    # set up camera input

    with pymba.Vimba() as vimba:

        # get system object

        system, camera, frame = init_camera(vimba)
        im = get_frame(camera, frame)
        H,W = im.shape

        # initialize data
        L = 40
        T = np.arange(L)
        D = np.zeros(L)
        X = [W//2]
        Y = [H//2]
        XP,YP = circle(X[0],Y[0],200)

        # set up plots
        fig = plt.figure(figsize=(12,7))
        gs = plt.GridSpec(4,1)

        fig.subplots_adjust(hspace=0.0)
        fig.suptitle("Testing")

        ax = list()
        ax.append(plt.subplot(gs[0,:]))
        ax.append(plt.subplot(gs[1:,:]))

        # diameter plot
        data = ax[0].plot(T, D, 'k')[0]
        minimum = ax[0].plot(T, D.min() * np.ones(D.shape), 'b')[0]

        ax[0].set_ylim(0, 100)
        ax[0].xaxis.set_visible(False)

        # image plot
        image = ax[1].imshow(im,
                             cmap=plt.get_cmap('spectral'),
                             interpolation='nearest')
        center = ax[1].plot(X,Y,'r+',ms=20, mew=1)[0]
        edge = ax[1].plot(XP,YP,'r-', lw=1)[0]


        ax[1].xaxis.set_visible(False)
        ax[1].yaxis.set_visible(False)
        ax[1].set_xlim(0,W)
        ax[1].set_ylim(H,0)
        ax[1].set_aspect('equal')

        plt.draw()
        plt.pause(0.1)

        # as long as a matplotlib window is open
        D0 = 1
        X0 = X[0]
        Y0 = Y[0]

        while plt.get_fignums():
            # get next frame
            im = get_frame(camera, frame)

            try:
                # get beam statistics
                X0,Y0,D0 = analyze_beam(im)


            except ValueError, TypError:
                # if the analysis step runs into an error,
                # just ignore it and use the last result
                pass

            # update diameter plot
            D = np.roll(D,1)
            D[0] = D0

            X[0] = X0
            Y[0] = Y0
            XP,YP = circle(X0,Y0, D[0])

            # update the matplotlib window
            data.set_ydata(D)
            minimum.set_ydata(D.min() * np.ones(D.shape))
            center.set_xdata(X)
            center.set_ydata(Y)
            edge.set_xdata(XP)
            edge.set_ydata(YP)
            image.set_data(im)
            ax[1].set_xlim(X0-W0, X0+W0+1)
            ax[1].set_ylim(Y0+W0+1, Y0-W0)
            ax[0].set_ylim(0, D.max()*1.8)
            fig.suptitle('(X,Y)=({:3.1f}, {:3.1f})\nD={:2.2f}px'.format(X[0],Y[0],D[0]))

            plt.draw()
            plt.pause(0.1)

        plt.imsave('test.png', im,
                    cmap=plt.get_cmap('spectral'))
        cleanup_cam(camera)
