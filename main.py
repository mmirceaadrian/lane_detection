import cv2
import numpy as np


def func(x, a, b):
    return b + a * x


cam = cv2.VideoCapture('test.mp4')
width = 480
height = 240

ableft = []
left_top = (0, 0)
left_bottom = (0, 0)
right_top = (0, 0)
right_bottom = (0, 0)

xm1 = 0
ym1 = 0
xm2 = 0
ym2 = 0
xm3 = 0
ym3 = 0
xm4 = 0
ym4 = 0
b1 = 0
a1 = 0
b2 = 0
a2 = 0
ok = 1

while True:

    ret, frame = cam.read()

    # ret (bool): Return code of the `read` operation. Did we get an image or not?
    #             (if not maybe the camera is not detected/connected etc.)

    # frame (array): The actual frame as an array.
    #                Height x Width x 3 (3 colors, BGR) if color image.
    #                Height x Width if Grayscale
    #                Each element is 0-255.
    #                You can slice it, reassign elements to change pixels, etc.

    if ret is False:
        break

    # 2
    frame = cv2.resize(frame, (480, 240))
    # cv2.imshow('Original', frame)
    first_frame = frame.copy()

    # 3
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', frame)

    # 4
    blackFrame = np.zeros(frame.shape, dtype=np.uint8)

    upper_left = [int(0.4 * width), int(0.8 * height)]
    upper_right = [int(0.6 * width), int(0.8 * height)]
    lower_left = [int(0), int(height)]
    lower_right = [int(width), int(height)]

    trapez = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)
    cv2.fillConvexPoly(blackFrame, trapez, 1)
    cv2.imshow('Trapezoid', blackFrame * 255)
    frame = frame * blackFrame

    cv2.imshow('Road', frame)

    # 5
    topDown = np.zeros(frame.shape, dtype=np.uint8)
    trapez = np.float32(trapez)
    magic_matrix = \
        cv2.getPerspectiveTransform(trapez, np.array([(width, 0), (0, 0), lower_left, lower_right], dtype=np.float32))

    topDown = cv2.warpPerspective(frame, magic_matrix, (width, height))
    cv2.imshow('Top-Down', topDown)

    # 6
    topDown = cv2.blur(topDown, ksize=(5, 5))

    # 7
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    topDown = np.float32(topDown)
    sobelFrameV = cv2.filter2D(topDown, -1, sobel_vertical)
    sobelFrameH = cv2.filter2D(topDown, -1, sobel_horizontal)

    # sobelFrameV = cv2.convertScaleAbs(sobelFrameV)
    # sobelFrameH = cv2.convertScaleAbs(sobelFrameH)
    # cv2.imshow('sobelV', sobelFrameV)
    # cv2.imshow('sobelH', sobelFrameH)

    topDownFiltered = sobelFrameV * sobelFrameV + sobelFrameH * sobelFrameH
    topDownFiltered = np.sqrt(topDownFiltered)
    topDownFiltered = cv2.convertScaleAbs(topDownFiltered)
    cv2.imshow('Sobel', topDownFiltered)

    # 8
    ret, topDownFiltered = cv2.threshold(topDownFiltered, int(100), 255, cv2.THRESH_BINARY)
    cv2.imshow('Binarized', topDownFiltered)

    # 9
    topDownFilteredCopy = topDownFiltered.copy()
    topDownFilteredCopy[0:height, 0:(int(width * 0.05))] = 0
    topDownFilteredCopy[0:height, (int(width * 0.95)):width] = 0
    topDownFilteredCopy[(int(0.95 * height)):height, :] = 0
    cv2.imshow('Edge Removed', topDownFilteredCopy)

    topDownFilteredCopy = cv2.morphologyEx(topDownFilteredCopy, cv2.MORPH_OPEN, (3, 3))

    points = np.argwhere(topDownFilteredCopy > 10)

    right_x = []
    right_y = []
    left_x = []
    left_y = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    x3 = []
    y3 = []
    x4 = []
    y4 = []

    for i in points:
        if 440 // 2 < i[1] < 400:
            right_x.append(i[1])
            right_y.append(i[0])
            if i[0] < height // 2:
                x2.append(i[1])
                y2.append(i[0])
            else:
                x4.append(i[1])
                y4.append(i[0])
        if i[1] < 440 // 2:
            left_x.append(i[1])
            left_y.append(i[0])
            if i[0] < height // 2:
                x1.append(i[1])
                y1.append(i[0])
            else:
                x3.append(i[1])
                y3.append(i[0])

    # 10

    if len(x1):
        xm1 = int(sum(x1) / len(x1))
        ym1 = int(sum(y1) / len(y1))

    if len(x2):
        xm2 = int(sum(x2) / len(x2))
        ym2 = int(sum(y2) / len(y2))
    if len(x3):
        xm3 = int(sum(x3) / len(x3))
        ym3 = int(sum(y3) / len(y3))

    if len(x4):
        xm4 = int(sum(x4) / len(x4))
        ym4 = int(sum(y4) / len(y4))

    try:
        b1 = (ym3 * xm1 - ym1 * xm3) / (xm1 - xm3)
        a1 = (ym1 - b1) / xm1
    except:
        print('No x')

    try:
        b2 = (ym4 * xm2 - ym2 * xm4) / (xm2 - xm4)
        a2 = (ym2 - b2) / xm2
    except:
        print('No x')

    left_top = (int((-b1) / a1), 0)
    left_bottom = (int((240 - b1) / a1), 240)

    right_top = (int((-b2) / a2), 0)
    right_bottom = (int((240 - b2) / a2), 240)

    if right_top[0] > 300 and right_bottom[0] > 300:
        ok = 1
        if left_top[0] < 200 and left_bottom[0] < 200:
            ok = 1
        else:
            ok = 2
    else:
        ok = 2

    cv2.line(topDownFilteredCopy, right_top, right_bottom, (255, 0, 0), 10)
    cv2.line(topDownFilteredCopy, left_top, left_bottom, (255, 0, 0), 10)
    cv2.imshow('Lines', topDownFilteredCopy)

    """

    if left_x:
        ableft = np.polynomial.polynomial.polyfit(left_x, left_y, 1)
        # ableft, _ = curve_fit(func, left_x, left_y)
    if right_x:
        abright = np.polynomial.polynomial.polyfit(right_x, right_y, 1)
        # abright, _ = curve_fit(func, right_x, right_y)

    left_top_y = 0
    left_top_x = (left_top_y - ableft[0]) / ableft[1]

    left_bottom_y = height
    left_bottom_x = (left_bottom_y - ableft[0]) / ableft[1]

    right_top_y = 0
    right_top_x = (right_top_y - abright[0]) / abright[1]

    right_bottom_y = height
    right_bottom_x = (right_bottom_y - abright[0]) / abright[1]

    try:
        if -np.power(10, 8) < left_top_x < np.power(10, 8) and 0 < left_top_x < width // 2:
            left_top = int(abs(left_top_x)), int(left_top_y)
            # left_top = int(max(left_x)), int(left_top_y)
            # if abs(left_top_x - max(x1)) < 50:
            #    left_top = int(max(x1)), int(left_top_y)

        if -np.power(10, 8) < left_bottom_x < 10 ** 8 and 0 < left_bottom_x < width // 2:
            left_bottom = int(abs(left_bottom_x)), int(left_bottom_y)
            # if abs(left_bottom_x - max(x3)) < 50:
            #    left_bottom = int(max(x3)), int(left_bottom_y)

        if -np.power(10, 8) < right_top_x < np.power(10, 8) and width // 2 < right_top_x < width:
             right_top = int(right_top_x), int(right_top_y)
            # if abs(right_top_x - max(right_x)) < 30 or abs(right_top_x - min(right_x)) < 30:
            #    right_top = int(max(right_x)), int(right_top_y)
            # if abs(right_top_x - max(x2)) < 40:
            #    right_top = int(max(x2)), int(right_top_y)


        if -np.power(10, 8) < right_bottom_x < 10 ** 8 and width // 2 < right_bottom_x < width:
            right_bottom = int(right_bottom_x), int(right_bottom_y)
            # if abs(right_bottom_x - min(right_x)) < 30 or abs(right_bottom_x - max(right_x)) < 30:
            #    right_bottom = int(min(right_x)), int(right_bottom_y)
            # if abs(right_bottom_x - max(x4)) < 37:
            #    right_bottom = int(min(x4)), int(right_bottom_y)
    except:
        print('No x')

  

    cv2.line(topDownFilteredCopy, left_top, left_bottom, (255, 0, 0), 10)
    cv2.line(topDownFilteredCopy, right_top, right_bottom, (255, 0, 0), 10)

    cv2.imshow('Lines', topDownFilteredCopy)
   

    """
    cv2.imshow('Original', first_frame)

    # 11
    blackFrame2 = np.zeros(frame.shape, dtype=np.uint8)
    cv2.line(blackFrame2, left_top, left_bottom, (255, 0, 0), 3)
    magic_matrix2 = \
        cv2.getPerspectiveTransform(np.array([(width, 0), (0, 0), lower_left, lower_right], dtype=np.float32), trapez)

    result1 = cv2.warpPerspective(blackFrame2, magic_matrix2, (width, height))

    blackFrame3 = np.zeros(frame.shape, dtype=np.uint8)
    cv2.line(blackFrame3, right_top, right_bottom, (255, 0, 0), 3)
    result2 = cv2.warpPerspective(blackFrame3, magic_matrix2, (width, height))

    result = result1 + result2
    cv2.imshow('result', result)

    leftWhite = np.argwhere(result1 > 10)
    rightWhite = np.argwhere(result2 > 10)

    final = first_frame.copy()

    trapez = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    trapez2 = np.array([
        (rightWhite[0][1], rightWhite[0][0]),
        (leftWhite[0][1], leftWhite[0][0]),
        (leftWhite[-1][1], leftWhite[-1][0]),
        (rightWhite[-1][1], rightWhite[-1][0])],
        dtype=np.int32)

    blackFrame5 = np.zeros(frame.shape, dtype=np.uint8)
    cv2.fillConvexPoly(blackFrame5, trapez2, 1)
    cv2.imshow('Trapezoid_Final', blackFrame5 * 255)

    trp = np.argwhere(blackFrame5 > 0)

    for i in trp:
        final[i[0]][i[1]][ok] = 150

    for i in leftWhite:
        final[i[0]][i[1]][0] = 50
        final[i[0]][i[1]][1] = 50
        final[i[0]][i[1]][2] = 250

    for i in rightWhite:
        final[i[0]][i[1]][0] = 50
        final[i[0]][i[1]][1] = 50
        final[i[0]][i[1]][2] = 250

    cv2.imshow('Final', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
