import sys
import time
import cv2
import numpy as np
import math

colours = [ (r,g,b) for r in [0,128,255] for g in [0,128,255] for b in [0,128,255] ]
c2 = 300
c6 = 900
open_kernel = np.ones((5, 5))

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

def locate_center_points(image, stats):
    line = np.zeros(len(image[0]), dtype=np.float32)
    
    for x in range(stats[0], stats[0]+stats[2]):
        exit_line = False
        enter_line = False
        y = stats[1]
        line_thickness = 0
        
        while not enter_line:
            if image[y][x] == 255:
                enter_line = True
            y += 1
            if enter_line:
                break
            if y >= stats[1] + stats[3]:
                exit_line = True
                enter_line = True
                y = None
                
        while not exit_line:
            if y + line_thickness >= stats[1] + stats[3] or image[y + line_thickness][x] == 0:
                exit_line = True
                break
            line_thickness += 1
        
        if y != None:
            line[x] = math.floor(y + line_thickness / 2)
        else:
            line[x] = math.nan
    
    return line

def height_to_freq(y,min_freq,max_freq):
    if y >= 0.99:
        return 0
        
    ratio = 2**(1/12)
    scale = math.log((max_freq/min_freq), ratio)
    return min_freq*2**(y*scale/12)
    
