# Program
import math
import numpy as np

c2 = 65.41
c6 = 1046.5


def height_to_freq(y,min_freq,max_freq):
    ratio = 2**(1/12)
    scale = math.log((max_freq/min_freq), ratio)
    return min_freq*2**(y*scale/12)


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
            if y > len(image):
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
        

#what frequency is half way between c2 and c6?
print(height_to_freq(0.5,c2,c6))
