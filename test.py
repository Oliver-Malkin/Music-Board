import math, numpy
import matplotlib.pyplot as plt

whole_time = 10 # self.duration
notes = [] # line[1]
for x in range(0, 20):
    for y in range(0, 5):
        notes.append(x)
a = 1

def interp(a, b, t):
    return a*(1-t) + b*t

def get_time(time):

    duration = whole_time/(len(notes)-1)
    f = (time%whole_time)/duration
    index = math.floor(f)%len(notes)
    fraction = (f)-index

    c = 1
    a = notes[index]
    b = notes[index+c]

    try:
        while a == b:
            c += 1
            b = notes[index+c]
    except IndexError:
        b = notes[len(notes)-1]
    
    for i in range(1, c):
        notes[index+i] = a+i*(b-a)/c

    fraction = fraction/c
    frequency = interp(a, b, fraction)

    return round(frequency, 3)

notes_ = []
for x in numpy.arange(0, 2, 0.01):
    notes_.append(get_time(x))
print(numpy.arange(0, 2, 0.01))

plt.scatter(numpy.arange(0, 2, 0.01), notes_, s=0.1)
plt.show()