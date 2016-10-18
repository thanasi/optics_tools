from pymba import *
import time

with Vimba() as vimba:
        # get system object
        system = vimba.getSystem()

        # list available cameras (after enabling discovery for GigE cameras)
        if system.GeVTLIsPresent:
            system.runFeatureCommand("GeVDiscoveryAllOnce")
            time.sleep(0.2)
        cameraIds = vimba.getCameraIds()
        for cameraId in cameraIds:
            print 'Camera ID:', cameraId

        # get and open a camera
        camera0 = vimba.getCamera(cameraIds[0])
        camera0.openCamera()

        # list camera features
        cameraFeatureNames = camera0.getFeatureNames()
        # for name in cameraFeatureNames:
        #     print 'Camera feature:', name

        # get the value of a feature
        print "Acquisition Mode: ", camera0.AcquisitionMode
        print "Exposure Mode: ", camera0.ExposureMode
        print "Exposure Time: ", camera0.ExposureTimeAbs
        print "Gain: ", camera0.Gain
        print "Iris Mode: ", camera0.IrisMode
        print "Pixel Format: ", camera0.PixelFormat
        print "Sensor Bits: ", camera0.SensorBits
        print "Height Max: ", camera0.HeightMax
        print "Width Max: ", camera0.WidthMax
        print "Sensor Info:"
        print "\tType: ", camera0.SensorType
        print "\tHeight: ", camera0.SensorHeight
        print "\tWidth: ", camera0.SensorWidth

        # set the value of a feature
        camera0.AcquisitionMode = 'SingleFrame'

        # create new frames for the camera
        frame0 = camera0.getFrame()    # creates a frame
        frame1 = camera0.getFrame()    # creates a second frame

        # announce frame
        frame0.announceFrame()

        # capture a camera image
        camera0.startCapture()
        frame0.queueFrameCapture()
        camera0.runFeatureCommand('AcquisitionStart')
        camera0.runFeatureCommand('AcquisitionStop')
        frame0.waitFrameCapture()

        # get image data...
        imgData = frame0.getBufferByteData()

        # clean up after capture
        camera0.endCapture()
        camera0.revokeAllFrames()

        # close camera
        camera0.closeCamera()
