import sys
import time
import cv2
import numpy as np
import math

def calibrate(vc):
    corners = []
    rval, frame = vc.read()
    
    def handleClick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(corners) < 4:
            corners.append((x, y))
            
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            
            if len(corners) > 1:
                cv2.line(frame, corners[-1], corners[-2], (0, 0, 255), 2)
            if len(corners) == 4:
                cv2.line(frame, corners[0], corners[3], (0, 0, 255), 2)

            cv2.imshow("Calibrate", frame)
    
    cv2.namedWindow("Calibrate")
    cv2.setMouseCallback("Calibrate", handleClick)
    cv2.imshow("Calibrate", frame)
    cv2.waitKey(1)
    
    while len(corners) < 4:
        cv2.waitKey(1)
    
    cv2.imshow("Calibrate", frame)
    print("got corners")
    
    
    cv2.destroyWindow("Calibrate")
    
    return corners

def makePerspectiveTransform(c):
    height = int(math.sqrt((c[2][0] - c[1][0]) * (c[2][0] - c[1][0]) + (c[2][1] - c[1][1]) * (c[2][1] - c[1][1])))
    width = height * 2
    points2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    M = cv2.getPerspectiveTransform(np.asarray(c, dtype=np.float32), points2)
    return (M, width, height)

def locate_center_points(image):
    line = np.zeros(len(image[0]))
    for x in range(0, len(image[0])-1):
        exit_line = False
        enter_line = False
        y = 0
        line_thickness = 0

        while not enter_line:
            if image[y][x] == 255:
                enter_line = True
            y += 1
            if y >= len(image):
                exit_line = True
                enter_line = True
                y = None

        while not exit_line:
            if image[y][x] == 0:
                exit_line = True
            line_thickness += 1
            if y + line_thickness > len(image):
                exit_line = True
                line_thickness -= 1
        
        if y != None:
            line[x] = math.floor(y + line_thickness/2)
        else:
            line[x] = None
    
    return line

def main():
    cv2.namedWindow("Preview")
    
    if len(sys.argv) > 1:
        vc = cv2.VideoCapture(int(sys.argv[1]))
    else:
        vc = cv2.VideoCapture(0)
    
    if vc.isOpened():
        rval, frame = vc.read()
        rval, frame = vc.read()
        time.sleep(1)
    else:
        print("Could not open webcam stream")
        return
    
    corners = [(61, 23), (596, 36), (579, 444), (67, 447)]#calibrate(vc)
    print(corners)
    print("calibrated")
    (M, width, height) = makePerspectiveTransform(corners)
    print("width, height = (%d, %d)" % (width, height))
    
    five = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    
    while rval:
        rval, frame = vc.read()
        corrected = cv2.warpPerspective(frame, M, (width, height))
        blur = cv2.GaussianBlur(corrected, (5, 5), 0)
        b, g, r = cv2.split(blur)
        grey = 255 - cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
            th, 8, cv2.CV_32S
        )
        #cv2.imshow("Preview", np.concatenate((grey, th)))
        #cv2.imshow("Preview", labels.astype("uint8") * 50)
        components = []
        lines = []
        for label in range(1, num_labels+1):
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[labels==label] = 255
            lines.append(locate_center_points(mask))
            components.append(cv2.bitwise_and(blur, blur, mask=mask))
        cv2.imshow("Preview", np.concatenate(tuple(components)))
        cv2.waitKey(1)
    
    vc.release()
    cv2.destroyWindow("Preview")
    
if __name__ == "__main__":
    main()