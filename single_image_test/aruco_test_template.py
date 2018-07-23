import numpy as np
import cv2
import pickle
import glob
import cv2.aruco as aruco


# This code is developed from aruco_detection_test, but adding the pose estimation.
# The first section is calibration, the second section is marker recognition and
# pose estimation.


###########################
# SECTION 1 - CALIBRATION #
##########################

objpoints = []  # 3D point in real world space where chess squares are
imgpoints = []  # 2D point in image plane, determined by CV2

CHESSBOARD_CORNERS_ROWCOUNT = 9     # Don't change as preset cal is 9x6
CHESSBOARD_CORNERS_COLCOUNT = 6

objp = np.zeros((CHESSBOARD_CORNERS_ROWCOUNT * CHESSBOARD_CORNERS_COLCOUNT, 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_CORNERS_ROWCOUNT, 0:CHESSBOARD_CORNERS_COLCOUNT].T.reshape(-1, 2)

images = glob.glob('img*.png')  # The calibration images
imageSize = None

for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    board, corners = cv2.findChessboardCorners(gray, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), None)

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

        img = cv2.drawChessboardCorners(img, (CHESSBOARD_CORNERS_ROWCOUNT, CHESSBOARD_CORNERS_COLCOUNT), corners_acc,
                                        board)

        cv2.imshow('Chessboard', img)   # GUI window that should pop-up
        cv2.waitKey(0)
    else:
        print("Not able to detect a chessboard in image: {}".format(iname))

cv2.destroyAllWindows()

if len(images) < 1:
    print(
        "Calibration was unsuccessful. No images of chessboards were found. Add images of chessboards and use or alter the naming conventions used in this file.")
    exit()

if not imageSize:

    print(
        "Calibration was unsuccessful. We couldn't detect chessboards in any of the images supplied. Try changing the patternSize passed into findChessboardCorners(), or try different pictures of chessboards.")
    exit()

calibration, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=imageSize,
    cameraMatrix=None,
    distCoeffs=None)    # Calibration coefficients

print(cameraMatrix)
print(distCoeffs)

f = open('calibration.pckl', 'wb')  # The pickle file that the calibration information is written to
pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
f.close()
print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))


#####################################################
# SECTION 2 - MARKER AQUISITION AND POSE ESTIMATION #
####################################################

images_2 = glob.glob('aru*.png')    # The single image of the aruco marker
imageSize = None
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)

for iname_2 in images_2:
    img_2 = cv2.imread(iname_2)
    cv2.imshow('window', img_2)
    gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

while (True):

        res = cv2.aruco.detectMarkers(gray, dictionary)
        #   print(res[0],res[1],len(res[2]))
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, dictionary, parameters=parameters)
        font = cv2.FONT_HERSHEY_SIMPLEX  # font for displaying text (below)

        if len(res[0]) > 0:
            cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

        if np.all(ids != None):

            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix,
                                                            distCoeffs)  # remember to change marker length (in meters)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error

            # Extracting the distance from marker ID

            list_tvec = list(tvec[0][0])

            # Scaling tvec according to camera matrix
            scaled_distance = (list_tvec[2]) / 2.07
            print("Distance from camera to marker is: ", list_tvec[2])

            for i in range(0, ids.size):
                aruco.drawAxis(img_2, cameraMatrix, distCoeffs, rvec[i], tvec[i], 0.1)  # Draw Axis
            aruco.drawDetectedMarkers(img_2, corners)  # Draw A square around the markers

            ###### DRAW ID #####
            strg = ''
            for i in range(0, ids.size):
                strg += str(ids[i][0]) + ', '

            cv2.putText(img_2, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            ##### DRAW "NO IDS" #####
            cv2.putText(img_2, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the resulting frame
        cv2.imshow('iname_2', img_2)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()
