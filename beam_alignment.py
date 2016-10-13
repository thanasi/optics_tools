import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from skimage.feature import canny
from skimage import color

mpl.use("MacOSX")

CAMERA_NUM = 0

def circle(x0,y0,r,num_points=1000):
    theta = np.arange(0,2*np.pi, 2*np.pi/num_points)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0

    return x,y

def analyze_beam(frame):

    X = 0
    Y = 0
    D = np.random.rand()*300

    return X,Y,D

if __name__ == "__main__":

    # set up camera input
    cap = cv2.VideoCapture(CAMERA_NUM)
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    H,W = frame.shape

    try:
        assert ret, "Error initializing camera"
    except:
        cap.release()
        raise


    # initialize data
    L = 200
    T = np.arange(L)
    D = np.zeros(L)
    X = [W//2]
    Y = [H//2]
    XP,YP = circle(X[0],Y[0],200)

    # set up plots
    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(4,1)

    fig.subplots_adjust(hspace=0.0)
    fig.suptitle("Testing")


    ax = list()
    ax.append(plt.subplot(gs[0,:]))
    ax.append(plt.subplot(gs[1:,:]))

    # diameter plot
    data = ax[0].plot(T, D, 'k')[0]
    minimum = ax[0].plot(T, D.min() * np.ones(D.shape), 'b')[0]

    ax[0].set_ylim(0, 350)
    ax[0].xaxis.set_visible(False)

    # image plot
    image = ax[1].imshow(frame, cmap=plt.get_cmap('gray'))
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

    while plt.get_fignums():
        # get next frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # get beam statistics
        X0,Y0,D0 = analyze_beam(frame)

        # update diameter plot
        D = np.roll(D,1)
        D[0] = D0

        # X[0] = X0
        # Y[0] = Y0
        XP,YP = circle(X[0], Y[0], D[0])

        # update the matplotlib window
        data.set_ydata(D)
        minimum.set_ydata(D.min() * np.ones(D.shape))
        center.set_xdata(X)
        center.set_ydata(Y)
        edge.set_xdata(XP)
        edge.set_ydata(YP)
        image.set_data(frame)
        fig.suptitle('(X,Y)=({:d}, {:d})\nD={:2.2f}px'.format(X[0],Y[0],D[0]))

        plt.draw()
        plt.pause(0.1)
