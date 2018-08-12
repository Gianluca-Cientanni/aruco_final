import numpy as np
import cv2
import pickle
import glob
import cv2.aruco as aruco
import time

# This code is developed from aruco_detection_test, but adding the pose estimation.
# The first section is calibration, the second section is marker recognition and
# pose estimation. Here you will need to take your calibration images yourself and
# save them as img1, img2, img3 etc.

###########################
# SECTION 1 - CALIBRATION #
##########################

objpoints = []
imgpoints = []

chessboard_row = 9
chessboard_col = 6

objp = np.zeros((chessboard_row * chessboard_col, 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_row, 0:chessboard_col].T.reshape(-1, 2)

images = glob.glob('img*.png')  # Calibration images
imageSize = None

for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    board, corners = cv2.findChessboardCorners(gray, (chessboard_row, chessboard_col), None)

    if board == True:
        objpoints.append(objp)

        corners_acc = cv2.cornerSubPix(
            image=gray,
            corners=corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30,
                      0.001))  # Last parameter is about termination critera
        imgpoints.append(corners_acc)

        if not imageSize:
            imageSize = gray.shape[::-1]

        img = cv2.drawChessboardCorners(img, (chessboard_row, chessboard_col), corners_acc,
                                        board)

        cv2.imshow('Chessboard', img)
        cv2.waitKey(0)
    else:
        print("Not able to detect a chessboard in image")

cv2.destroyAllWindows()

if len(images) < 1:
    print(
        "Calibration was unsuccessful.")
    exit()

if not imageSize:

    print(
        "Calibration was unsuccessful.")
    exit()

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=imageSize,
    cameraMatrix=None,
    distCoeffs=None)

print(cameraMatrix)
print(distCoeffs)

f = open('calibration.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()
print('Calibration successful.')

#####################################################
# SECTION 2 - MARKER AQUISITION AND POSE ESTIMATION #
####################################################

cap = cv2.VideoCapture(0)   # The argument of VideoCapture is the port number that the camera you wish to use
                            # is occupying. The standard is 0 and should be sufficient if no other cameras
                            # are being used by the computer.

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res = cv2.aruco.detectMarkers(gray,dictionary)
    #   print(res[0],res[1],len(res[2]))
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

    if len(res[0]) > 0:
        cv2.aruco.drawDetectedMarkers(gray,res[0],res[1])
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if np.all(ids != None):

        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix,
                                                        distCoeffs) # remember to change marker length
        (rvec-tvec).any() # get rid of that nasty numpy value array error

        # Extracting the distance from marker ID

        list_tvec = list(tvec[0][0])

        # Scaling tvec according to camera matric
        scaled_distance = (list_tvec[2])/2.07
        print("Distance from camera to marker is: " , list_tvec[2])

        for i in range(0, ids.size):
            aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec[i], tvec[i], 0.1)  # Draw Axis
        aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

        ###### DRAW ID #####
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0]) + ', '

        cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    else:
        ##### DRAW "NO IDS" #####
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(8)

cap.release()
cv2.destroyAllWindows()
