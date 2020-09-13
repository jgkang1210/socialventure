#-*-coding:utf-8-*-

#Marker Data base
import cv2 as cv
from cv2 import aruco
import numpy as np
import glob
import time

class markerNode:
    """
    Marker Node which contains data
    """
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.next = None


class markerMap:
    """
    Map data for every marker on single image
    """
    def __init__(self, size):
        self.size  = size


class markerAggregation:
    """
    Bind the near markers into one aggregated object
    """
    def __init__(self, size, id, x, y, north = None, south = None, west = None, east = None):
        self.size = size
        self.id = id
        self.x = x
        self.y = y
        self.northConnectedNode = north
        self.southConnectedNode = south
        self.westConnectedNode = west
        self.eastConnectedNode = east
        self.next = None

    def update(self, x, y, north = None, south = None, west = None, east = None):
        self.x = x
        self.y = y
        self.northConnectedNode = north
        self.southConnectedNode = south
        self.westConnectedNode = west
        self.eastConnectedNode = east
        self.next = None

# We use 250 ID
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# 1~4는 그냥 카메라 캘리브레이션을 위해 제작한 코드이므로 딱히 신경쓸 필요는 없을듯.
# 1. Once it generate the board we print the board and take with our camera.
def generateCharucoBoard():
    board = aruco.CharucoBoard_create(7, 5, 0.04, 0.02, aruco_dict)
    imboard = board.draw((2000, 2000))
    cv.imwrite('charuco_board.tiff', imboard)
    return board

# 2. If press 'c' capture current image from camrera, If press 'esc' finish the task
def getCharucoBoardImgfromCamera():
    print("press esc to terminate")
    print("press c to capture")
    cap = cv.VideoCapture(-1)
    i = 1 #step

    while (1):
        # Take each frame
        _, frame = cap.read()
        # Convert BGR to HSV
        frame = cv.resize(frame, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC)

        cv.imshow('output', frame)

        k = cv.waitKey(5) & 0xFF
        #press esc to terminate
        if k == 27:
            break

        #press c to capture
        #save our pictures in folder "calibpic"
        if k == ord('c'):
            workdir = "./calibpic/"
            cv.imwrite(workdir + 'calibBoard'+str(i)+'.jpg', frame)
            print('image' + str(i) + 'saved')
            i = int(i) + 1

    cv.destroyAllWindows()

# 3. We calibrate our camera to get camera matrix and distorsion vector
def calibrateFromCharucoBoardImage(board):
    #get pictures in folder "calibpic"
    workdir = "./calibpic/"
    images = glob.glob(workdir + '*.jpg')
    print(images)
    print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0

    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        print("=> Processing image {0}".format(im))
        frame = cv.imread(im)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict)
        fixedcorners = []

        # If we detect the corners from image
        if len(corners) > 0:
            # SUB PIXEL DETECTION
            for corner in corners:
                corner2 = cv.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                fixedcorners.append(corner2)

            # InterpolateCornersCharuco return [1]:corners, [2]:Ids
            CornerandIds = aruco.interpolateCornersCharuco(fixedcorners, ids, gray, board)
            if CornerandIds[1] is not None and CornerandIds[2] is not None and len(CornerandIds[1]) > 3 and decimator % 1 == 0:
                allCorners.append(CornerandIds[1])
                allIds.append(CornerandIds[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize

# 4. calibrate and get the vectors
def calibrate_camera(allCorners,allIds,imsize, board):
    """
    Calibrates the camera using the dected corners.
    """
    print("CAMERA CALIBRATION")

    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv.CALIB_USE_INTRINSIC_GUESS + cv.CALIB_RATIONAL_MODEL + cv.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv.TERM_CRITERIA_EPS & cv.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

#5. Find marker on screen
def findMarkerOnScreen(mtx, dist, rvecs, tvecs):
    """
    Scan all marker data on the screen and made map
    :return: map data
    """
    start = time.time()
    cap = cv.VideoCapture(-1)
    i = 1  # step

    while (1):
        # wait for 1 seconds
        if time.time() - start > 1:
            break

        # Take each frame
        _, frame = cap.read()
        # Convert BGR to HSV
        frame = cv.resize(frame, None, fx=1, fy=1, interpolation=cv.INTER_CUBIC)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        # SUB PIXEL DETECTION
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)
        fixedCorners = []

        for corner in corners:
            corner2 = cv.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
            fixedCorners.append(corner2)

        frame_markers = aruco.drawDetectedMarkers(frame.copy(), fixedCorners, ids)

        size_of_marker = 0.0285  # side lenght of the marker in meter
        rvecs, tvecs, _obj = aruco.estimatePoseSingleMarkers(fixedCorners, size_of_marker, mtx, dist)

        if tvecs is None:
            cv.imshow('output', frame)
            print("-----------------------------")
            k = cv.waitKey(10) & 0xFF
            # press esc to terminate
            if k == 27:
                break
            continue

        cv.imshow('output', frame_markers)

        k = cv.waitKey(10) & 0xFF
        # press esc to terminate
        if k == 27:
            break

    cv.destroyAllWindows()

    return ids, fixedCorners, rvecs, tvecs

def processMarker(ids, fixedCorners, rvecs, tvecs):
    """
    1. [number of markers],[x value],[y value],[z value]

    process to

    2. [number of markers][average of x value][average of y value][average of z value]

    :param ids: all ids that recgonzise from outside
    :param fixedCorners: corner point of each markers
    :param rvecs: all rotational vectors of each markers
    :param tvecs: all translational vectors of each markers

    :return: return the aggregated marker which contain x, y, z value
    """

    # [number of markers][average of x value][average of y value][average of z value]
    # all initialized to zero
    aggregatedMarker = np.zeros((11,4),np.float32)

    length = ids.size
    for x in range(length):
        currentId = ids[x][0]
        aggregatedMarker[currentId][0] += 1
        aggregatedMarker[currentId][1] += tvecs[x][0][0]
        aggregatedMarker[currentId][2] += tvecs[x][0][1]
        aggregatedMarker[currentId][3] += tvecs[x][0][2]

    for x in range(11):
        if aggregatedMarker[x][0] != 0:
            aggregatedMarker[x][1] /= aggregatedMarker[x][0]
            aggregatedMarker[x][2] /= aggregatedMarker[x][0]
            aggregatedMarker[x][3] /= aggregatedMarker[x][0]

    return aggregatedMarker


class currentMarkerLocation:
    """
    Save all the marker location data to calculate the speed of marker.
    """
    def __init__(self, prev = None, cur = None):
        self.prev = prev
        self.cur = cur

    def updateLocation(self, cur):
        self.prev = self.cur
        self.cur = cur

    def calculateMarkerSpeed(self):
        prevLoc = self.prev
        currentLoc = self.cur
        speed_vector = np.zeros((12, 3), np.float32)

        if prevLoc is None:
            return speed_vector

        for x in range(11):
            prev_x_data = prevLoc[x][1]
            cur_x_data = currentLoc[x][1]
            prev_y_data = prevLoc[x][2]
            cur_y_data = currentLoc[x][2]
            prev_z_data = prevLoc[x][3]
            cur_z_data = currentLoc[x][3]
            speed_vector[x][0] = cur_x_data - prev_x_data
            speed_vector[x][1] = cur_y_data - prev_y_data
            speed_vector[x][2] = cur_z_data - prev_z_data
            speed_vector[11][0] += speed_vector[x][0]
            speed_vector[11][1] += speed_vector[x][1]
            speed_vector[11][2] += speed_vector[x][2]

        for x in range(3):
            speed_vector[11][x] /= 10

        return speed_vector

    def calculateDiff(self, speed_vector):
        max_diff = 0
        max_diff_id = 1

        prevLoc = self.prev
        currentLoc = self.cur

        if prevLoc is None:
            return None

        for x in range(11):
            if x != 1:
                current_diff = abs(speed_vector[x][0]-speed_vector[11][0])
                current_diff += abs(speed_vector[x][1]-speed_vector[11][1])
                current_diff += abs(speed_vector[x][2] - speed_vector[11][2])
                if current_diff > max_diff:
                    max_diff_id = x
                    max_diff = current_diff

        if max_diff < 0.1:
            return None

        return max_diff_id


    def printLocation(self, markerDatabase):
        currentLoc = self.cur
        stringList = "현재 보이는 물체는 "

        for x in range(11):
            if currentLoc[x][0] != 0:
                name = markerDatabase.findObjectById(x-1)
                stringList = stringList + name + ", "

        stringList = stringList + "입니다!"
        return stringList

    def printGrabbedObject(self, markerDatabase,id):
        stringList = markerDatabase.findObjectById(id-1) + "를 잡으셨습니다!"
        return stringList


class markerGraph:
    """
    Save Recognized markers
    """
    def __init__(self, markerNode = None):
        self.head = markerNode
        self.routeLength = 1

    def insertFirstNode(self, firstNode):
        new_node = firstNode
        temp_node = self.head
        self.head = new_node
        self.head.next = temp_node
        self.routeLength += 1

    def insertLast(self, insertNode):
        node = self.head
        while True:
            if node.next == None:
                break
            node = node.next

        new_node = insertNode
        node.next = new_node
        self.routeLength += 1

    def selectNode(self, num):
        if self.routeLength < num:
            print("Overflow")
            return
        node = self.head
        count = 0
        while count < num:
            node = node.next
            count += 1
        return node

    def deleteHead(self):
        node = self.head
        self.head = node.next
        del node
        self.routeLength -= 1

    def length(self):
        return str(self.routeLength)


class markerDatabase:
    """
    Marker Database
    marker ID  :  1~10
    ID : 1 --> Bottle
    ID : 2 --> Book : 실감나게 배우는 제어공학
    ID : 3 --> Book2 : 모두의 라즈베리 파이
    ID : 4 --> Cap
    ID : 5 --> Flower pot
    ID : 6 --> Table Corner
    ID : 7 --> Purifier
    ID : 8 --> Fire extinguisher
    ID : 9 --> Door
    ID : 10 --> Calendar
    """
    def __init__(self, markerDatabase = None):
        self.markerDatabase = markerDatabase

    def initialize(self):
        self.markerDatabase = markerGraph(markerNode("물병", 1))
        self.markerDatabase.insertLast(markerNode("책 실감나게 배우는 제어공학", 2))
        self.markerDatabase.insertLast(markerNode("책 모두의 라즈베리파이", 3))
        self.markerDatabase.insertLast(markerNode("모자", 4))
        self.markerDatabase.insertLast(markerNode("꽃병", 5))
        self.markerDatabase.insertLast(markerNode("책상 모서리", 6))
        self.markerDatabase.insertLast(markerNode("정수기", 7))
        self.markerDatabase.insertLast(markerNode("소화기", 8))
        self.markerDatabase.insertLast(markerNode("문", 9))
        self.markerDatabase.insertLast(markerNode("달력", 10))


    def findObjectById(self, id):
        """
        Give id to get the information of the object
        :param id: id of the object
        :return: Name of the object
        """
        return self.markerDatabase.selectNode(id).name


# Send to server
def html_init():
    str1 = "<!DOCTYPE html><html><head><title>hello</title></head><body></body></html>"
    html_file = open('/home/pi/django/subuway/subuweb/templates/subuweb/index.html')
    html_file.write(str1)
    html_file.close()

# Send to server
def html_sending(s):
    str2 = s
    str3 = "<!DOCTYPE html><html lang=" + "kr" + "><head><meta charset=" + "UTF-8>" + "<title>hello</title></head><body><h2>" + str2 + "</h2></body></html>"
    html_file = open('/home/pi/django/subuway/subuweb/templates/subuweb/index.html')
    html_file.write(str3)
    html_file.close()

#main function
if __name__ == '__main__':
    #create marker database
    html_init()
    markerDatabase = markerDatabase()
    markerDatabase.initialize()

    #initial setting for camera
    charucoBoard = generateCharucoBoard()
    getCharucoBoardImgfromCamera()
    allCorners, allIds, imsize = calibrateFromCharucoBoardImage(charucoBoard)
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(allCorners, allIds, imsize, charucoBoard)
    currentMarkerLocation = currentMarkerLocation(None)

    #html setting
    html_sending("")

    step = 0
    while(1):
        ids, fixedCorners, rvecs, tvecs = findMarkerOnScreen(mtx, dist, rvecs, tvecs)

        k = cv.waitKey(500) & 0xFF
        # press esc to terminate

        if k == 27:
            break

        if ids is None:
            print("Can't recognize")
            continue

        aggregatedMarker = processMarker(ids, fixedCorners, rvecs, tvecs)
        currentMarkerLocation.updateLocation(aggregatedMarker)
        stringList = currentMarkerLocation.printLocation(markerDatabase)
        print(stringList)
        html_sending(stringList)
        speed_vector = currentMarkerLocation.calculateMarkerSpeed()

        grabbed_object = currentMarkerLocation.calculateDiff(speed_vector)

        if grabbed_object is None:
            continue
        else:
            stringList = currentMarkerLocation.printGrabbedObject(markerDatabase, grabbed_object)
            print(stringList)
            html_sending(stringList)


        #k = cv.waitKey(3000) & 0xFF
        # press esc to terminate

        if k == 27:
            break

        #print(speed_vector)

        step += 1


