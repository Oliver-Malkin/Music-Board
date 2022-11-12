# Program
import math

c2 = 65.41
c6 = 1046.5

def height_to_freq(y,min_freq,max_freq):
    ratio = 2**(1/12)
    scale = math.log((max_freq/min_freq), ratio)
    return min_freq*2**(y*scale/12)

#what frequency is half way between c2 and c6?
print(height_to_freq(0.5,c2,c6))
