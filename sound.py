import math, time, array, numpy
from pysoundio import (
    PySoundIo,
    SoundIoFormatFloat32LE
)

class Player:
    def __init__(self, backend: int = None, device: int = None, rate: int = 44100, blocksize: int = 4096, duration: int = 10):
        self.pysoundio = PySoundIo(backend=backend)

        self.duration = duration # How long each line is in secions
        self.seconds_offset = 0.0
        self.seconds_per_frame = 1.0 / rate

        self.lines = [] # (type, array of frequencies)

        self.pysoundio.start_output_stream(
            device_id=device,
            channels=1,
            sample_rate=rate,
            block_size=blocksize,
            dtype=SoundIoFormatFloat32LE,
            write_callback=self.callback
        )

    def close(self):
        self.pysoundio.close()

    def sample(self, time):
        samp = 0
        for i in range(len(self.lines)):
            line = self.lines[i] # the current line

            duration = self.duration/(len(line[1])-1)
            f = (time%self.duration)/duration
            index = math.floor(f)%len(line[1])
            fraction = (f)-index

            a = line[1][index]
            b = line[1][index+1]
            frequency = a*(1-fraction)+(b*fraction)
            if line[0] == 'sin':
                radians = 2.0 * math.pi * frequency
                samp += math.sin(time * radians)
        return samp

    def callback(self, data, length):
        indata = array.array('f', [0] * length)
        for i in range(0, length):
            indata[i] = self.sample(self.seconds_offset + i * self.seconds_per_frame)
        data[:] = indata.tobytes()
        self.seconds_offset += self.seconds_per_frame * length

def main() -> None:
    player = Player(duration=10)
    player.lines = [('sin', [200, 210, 220, 240, 300, 400])]

    print('Playing...')
    print('CTRL-C to exit')

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('Exiting...')

    player.close()

if __name__ == '__main__':
    main()
