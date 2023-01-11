import numpy as np
import cv2 as cv2

# Load left and right images
imgL = cv2.imread("tsukuba_l.png", 0)
imgR = cv2.imread("tsukuba_r.png",0)

# cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
# cv2.resizeWindow('disp',600,600)

# Create window and trackbars for adjusting SGBM parameters
cv2.namedWindow('disparity')
cv2.createTrackbar('minDisparity', 'disparity', 0, 100, lambda x: None)
cv2.createTrackbar('numDisparities', 'disparity', 16, 256, lambda x: None)
cv2.createTrackbar('blockSize', 'disparity', 5, 21, lambda x: None)
cv2.createTrackbar('uniquenessRatio', 'disparity', 15, 100, lambda x: None)
cv2.createTrackbar('speckleWindowSize', 'disparity', 0, 200, lambda x: None)
cv2.createTrackbar('speckleRange', 'disparity', 2, 32, lambda x: None)
cv2.createTrackbar('disp12MaxDiff', 'disparity', 1, 100, lambda x: None)

sgbm = cv2.StereoSGBM_create()
while True:
    # Get current values of trackbars
    minDisparity = cv2.getTrackbarPos('minDisparity', 'disparity')
    numDisparities = cv2.getTrackbarPos('numDisparities', 'disparity')
    blockSize = cv2.getTrackbarPos('blockSize', 'disparity')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', 'disparity')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', 'disparity')
    speckleRange = cv2.getTrackbarPos('speckleRange', 'disparity')
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', 'disparity')

    # Create SGBM object
    
    sgbm.setMinDisparity(minDisparity)
    sgbm.setNumDisparities(numDisparities)
    sgbm.setBlockSize(blockSize)
    sgbm.setUniquenessRatio(uniquenessRatio)
    sgbm.setSpeckleWindowSize(speckleWindowSize)
    sgbm.setSpeckleRange(speckleRange)
    sgbm.setDisp12MaxDiff(disp12MaxDiff)

    # Compute disparity map
    disparity = sgbm.compute(imgL, imgR)

    # Normalize disparity map
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Show disparity map
    cv2.imshow('disparity', disparity_norm)
    
    # Exit loop if user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()