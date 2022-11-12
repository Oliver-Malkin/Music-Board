import math, time, array
from pysoundio import (
    PySoundIo,
    SoundIoFormatFloat32LE
)

class Player:
    def __init__(self, freq: float = 0, backend: int = None, device: int = None, rate: int = 44100, blocksize: int = 4096):
        self.pysoundio = PySoundIo(backend=backend)

        self.freq = freq
        self.seconds_offset = 0.0
        self.seconds_per_frame = 1.0 / rate

        self.lines = [] # (type, frequency)

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
            line = self.lines[i]
            if line[0] == 'sin':
                radians = 2.0 * math.pi * line[1]
                samp += math.sin(time * radians)
        return samp

    def callback(self, data, length):
        indata = array.array('f', [0] * length)
        for i in range(0, length):
            indata[i] = self.sample(self.seconds_offset + i * self.seconds_per_frame)
        data[:] = indata.tobytes()
        self.seconds_offset += self.seconds_per_frame * length

def main() -> None:
    player = Player(0)
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
