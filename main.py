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
    if ret is False:
        break

    frame = cv2.resize(frame, (width, height))
    first_frame = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', frame)

    blackFrame = np.zeros(frame.shape, dtype=np.uint8)

    # Trapezoid
    upper_left = [int(0.4 * width), int(0.8 * height)]
    upper_right = [int(0.6 * width), int(0.8 * height)]
    lower_left = [int(0), int(height)]
    lower_right = [int(width), int(height)]

    # Create trapezoid
    trapez = np.array([upper_right, upper_left, lower_left,
                      lower_right], dtype=np.int32)
    cv2.fillConvexPoly(blackFrame, trapez, 1)
    cv2.imshow('Trapezoid', blackFrame * 255)

    # remove everything outside trapezoid
    frame = frame * blackFrame

    cv2.imshow('Road', frame)

    # Create top-down view
    topDown = np.zeros(frame.shape, dtype=np.uint8)
    trapez = np.float32(trapez)
    magic_matrix = \
        cv2.getPerspectiveTransform(trapez, np.array(
            [(width, 0), (0, 0), lower_left, lower_right], dtype=np.float32))

    topDown = cv2.warpPerspective(frame, magic_matrix, (width, height))
    cv2.imshow('Top-Down', topDown)

    # Blur
    topDown = cv2.blur(topDown, ksize=(5, 5))

    # Sobel for edge detection
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])
    sobel_horizontal = np.transpose(sobel_vertical)

    topDown = np.float32(topDown)
    sobelFrameV = cv2.filter2D(topDown, -1, sobel_vertical)
    sobelFrameH = cv2.filter2D(topDown, -1, sobel_horizontal)

    # Apply Sobel
    topDownFiltered = sobelFrameV * sobelFrameV + sobelFrameH * sobelFrameH
    topDownFiltered = np.sqrt(topDownFiltered)
    topDownFiltered = cv2.convertScaleAbs(topDownFiltered)
    cv2.imshow('Sobel', topDownFiltered)

    # Binarize
    ret, topDownFiltered = cv2.threshold(
        topDownFiltered, int(100), 255, cv2.THRESH_BINARY)
    cv2.imshow('Binarized', topDownFiltered)

    # Remove edges
    topDownFilteredCopy = topDownFiltered.copy()
    topDownFilteredCopy[0:height, 0:(int(width * 0.05))] = 0
    topDownFilteredCopy[0:height, (int(width * 0.95)):width] = 0
    topDownFilteredCopy[(int(0.95 * height)):height, :] = 0
    cv2.imshow('Edge Removed', topDownFilteredCopy)

    # Remove noise
    topDownFilteredCopy = cv2.morphologyEx(
        topDownFilteredCopy, cv2.MORPH_OPEN, (3, 3))

    # Find points
    threshold = 10
    points = np.argwhere(topDownFilteredCopy > threshold)

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

    # Find points in each half
    for y, x in points:
        if 440 // 2 < x < 400:
            right_x.append(x)
            right_y.append(y)
            if y < height // 2:
                x2.append(x)
                y2.append(y)
            else:
                x4.append(x)
                y4.append(y)
        if x < 440 // 2:
            left_x.append(x)
            left_y.append(y)
            if y < height // 2:
                x1.append(x)
                y1.append(y)
            else:
                x3.append(x)
                y3.append(y)

    # Find average of points in each half
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

    # Find lines
    try:
        b1 = (ym3 * xm1 - ym1 * xm3) / (xm1 - xm3)
        a1 = (ym1 - b1) / xm1
    except:
        print('No x/y')

    try:
        b2 = (ym4 * xm2 - ym2 * xm4) / (xm2 - xm4)
        a2 = (ym2 - b2) / xm2
    except:
        print('No x/y')

    # Draw lines
    left_top = (int((-b1) / a1), 0)
    left_bottom = (int((240 - b1) / a1), 240)

    right_top = (int((-b2) / a2), 0)
    right_bottom = (int((240 - b2) / a2), 240)

    # Check if lines are in the middle
    if right_top[0] > 300 and right_bottom[0] > 300:
        ok = 1
        if left_top[0] < 200 and left_bottom[0] < 200:
            ok = 1
        else:
            ok = 2
    else:
        ok = 2

    # Draw lines
    cv2.line(topDownFilteredCopy, right_top, right_bottom, (255, 0, 0), 10)
    cv2.line(topDownFilteredCopy, left_top, left_bottom, (255, 0, 0), 10)
    cv2.imshow('Lines', topDownFilteredCopy)

    cv2.imshow('Original', first_frame)

    # Perspective transform
    blackFrame2 = np.zeros(frame.shape, dtype=np.uint8)
    cv2.line(blackFrame2, left_top, left_bottom, (255, 0, 0), 3)
    magic_matrix2 = \
        cv2.getPerspectiveTransform(np.array(
            [(width, 0), (0, 0), lower_left, lower_right], dtype=np.float32), trapez)

    result1 = cv2.warpPerspective(blackFrame2, magic_matrix2, (width, height))

    blackFrame3 = np.zeros(frame.shape, dtype=np.uint8)
    cv2.line(blackFrame3, right_top, right_bottom, (255, 0, 0), 3)
    result2 = cv2.warpPerspective(blackFrame3, magic_matrix2, (width, height))

    # Combine results
    result = result1 + result2
    cv2.imshow('result', result)

    # Find points
    leftWhite = np.argwhere(result1 > 10)
    rightWhite = np.argwhere(result2 > 10)

    final = first_frame.copy()

    # Draw lines
    trapez = np.array([upper_right, upper_left, lower_left,
                      lower_right], dtype=np.int32)

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

    # Draw trapezoid
    for x in trp:
        final[x[0]][x[1]][ok] = 150

    # Draw points
    for x in leftWhite:
        final[x[0]][x[1]][0] = 50
        final[x[0]][x[1]][1] = 50
        final[x[0]][x[1]][2] = 250

    for x in rightWhite:
        final[x[0]][x[1]][0] = 50
        final[x[0]][x[1]][1] = 50
        final[x[0]][x[1]][2] = 250

    cv2.imshow('Final', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
