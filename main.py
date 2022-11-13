import array, sys, time, cv2, math, pyaudio, tkinter
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from vision import *

freqs = []

def interp(a, b, t):
    return a*(1-t) + b*t

class Player:
    def __init__(self, duration: int = 5):
        p = pyaudio.PyAudio()
        rate = 20000
        self.stream = p.open(
            format=p.get_format_from_width(4),
            channels=1,
            rate=rate,
            input=False,
            output=True,
            stream_callback=self.callback,
            frames_per_buffer=64)
        print(self.stream)

        self.duration = duration # How long each line is in secions
        self.seconds_per_frame = 1.0 / rate

        self.lines = [] # (type, array of frequencies)
        
        self.stream.start_stream()

    def close(self):
        self.stream.stop_stream()
        self.stream.close()

    def sample(self, time):
        samp = 0
        #print("freqs:", end=" ")
        for i in range(len(self.lines)):
            (kind, func, minim, maxim, width) = self.lines[i] # the current line
            
            duration = self.duration/(width-1)
            f = (time%self.duration)/duration
            if f < minim or f > maxim:
                continue
            #index = math.floor(f)%len(line[1])
            
            frequency = func(f)
            #print(frequency, ", ", end="")
            
            freqs.append((f, frequency, time))
            match kind:
                case 'sin':
                    radians = 2.0 * math.pi * frequency
                    samp += math.sin(time * radians)
                case 'saw':
                    t = frequency*time
                    samp += t-math.floor(t)
                case 'square':
                    t = frequency*time
                    samp += 2*(2*t%2)-1

        return samp
    
    def load_lines(self, lines):
        print("lines:", len(lines))
        
        for (kind, line) in lines:
            xs = []
            ys = []
            
            for i in range(0, len(line), 16):
                #print(i, line[i])
                #val = sum(line[min(j, len(line)-1)] for j in range(i, i+16)) / 16.0
                if line[i] > 0:
                    xs.append(i)
                    ys.append(line[i])
            
            if len(xs) < 2:
                continue
            
            plt.scatter(xs, ys)
            
            f = interpolate.CubicSpline(xs, ys)
            plt.plot(xs, list(map(lambda x: f(x), xs)))
            
            self.lines.append((kind, f, xs[0], xs[-1], len(line)))

    def callback(self, in_data, frame_count, time_info, status):
        indata = array.array('f', [0] * frame_count)
        for i in range(frame_count):
            sample = self.sample(time_info["output_buffer_dac_time"] % self.duration + i * self.seconds_per_frame)
            indata[i] = sample
        return (bytes(indata), pyaudio.paContinue)
        #return (bytes([int(128 + 100 * math.sin(t * 0.035)) for t in range(frame_count * 2)]), pyaudio.paContinue)

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
    
    corners = [(61, 18), (590, 18), (568, 432), (73, 416)]#calibrate(vc)
    print(corners)
    print("calibrated")
    (M, width, height) = makePerspectiveTransform(corners)
    print("width, height = (%d, %d)" % (width, height))
    
    five = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    
    player = Player(duration=15)
    
    while rval and player.stream.is_active():
        print("starting frame")
        rval, frame = vc.read()
        corrected = cv2.warpPerspective(frame, M, (width, height))
        blur = cv2.GaussianBlur(corrected, (7, 7), 0)
        b, g, r = cv2.split(blur)
        grey = 255 - cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        #ret, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, th = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)
        #th = cv2.morphologyEx(th, cv2.MORPH_OPEN, open_kernel)
        (num_labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(
            th, 8, cv2.CV_32S
        )
        
        components = []
        lines = []
        
        pairs = zip(range(0, num_labels), stats)
        next(pairs)
        for (label, stat) in pairs:
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[labels==label] = 255
            r_avg = float(sum(map(sum, cv2.bitwise_and(r, r, mask=mask)))) / stat[4]
            g_avg = float(sum(map(sum, cv2.bitwise_and(g, g, mask=mask)))) / stat[4]
            b_avg = float(sum(map(sum, cv2.bitwise_and(b, b, mask=mask)))) / stat[4]
            if g_avg >= r_avg + 10 and g_avg >= b_avg + 10:
                colour = "green"
            elif r_avg >= g_avg + 15 and r_avg >= b_avg + 15:
                colour = "red"
            else:
                colour = "blue"
            lines.append((locate_center_points(mask, stat), colour))
            components.append(cv2.bitwise_and(blur, blur, mask=mask))
        
        for (line, colour_name) in lines:
            colour = {
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "blue": (255, 0, 0),
                }[colour_name]
            
            for x1 in range(width-1):
                x2 = x1 + 1
                y1 = line[x1]
                y2 = line[x2]
                if y1 > 0 and y2 > 0:
                    cv2.line(blur, (x1, int(y1)), (x2, int(y2)), colour, 3)
        
        for stat in stats:
            cv2.rectangle(blur, (stat[0], stat[1]), (stat[0] + stat[2], stat[1] + stat[3]), (255, 0, 0), 1)
        
        freq_lines = []
        for (line, colour_name) in lines:
            kind = {
                "red": "square",
                "green": "saw",
                "blue": "sin"
                }[colour_name]
            freq_lines.append((kind, [ height_to_freq(1 - y / height, c2, c6) for y in line ]))
        
        player.load_lines(freq_lines)
        
        cv2.imshow("Preview", blur)
        cv2.waitKey(player.duration * 1000)
        break
    
    vc.release()
    cv2.destroyWindow("Preview")
    player.close()
    
    plt.plot(list(map(lambda x: x[0], freqs)))
    plt.plot(list(map(lambda x: x[1], freqs)))
    plt.plot(list(map(lambda x: x[2], freqs)))
    plt.show()

if __name__ == '__main__':
    main()
