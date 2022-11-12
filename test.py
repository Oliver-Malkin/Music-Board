import math, numpy
import matplotlib.pyplot as plt

time1 = 20 # self.duration
#time = 5 # time 
notes = [1, 2, 4, 8, 16] # line[1]

def get_time(time):
    duration = time1/(len(notes)-1)
    f = (time%time1)/duration
    index = math.floor(f)%len(notes)
    fraction = (f)-index

    a = notes[index]
    b = notes[index+1]
    frequency = a*(1-fraction)+(b*fraction)

    return round(frequency, 3)

notes_ = []
for x in numpy.arange(0, 50, 0.01):
    notes_.append(get_time(x))

plt.scatter(numpy.arange(0, 50, 0.01), notes_, s=0.01)
plt.show()