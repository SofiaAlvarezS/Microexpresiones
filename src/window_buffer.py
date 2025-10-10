# src/window_buffer.py
from collections import deque

class SlidingWindowBuffer:
    def __init__(self, window_size=16, stride=8):
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size)
        self.counter = 0
        self.last_emit = -stride

    def add_frame(self, frame):
        """
        Añade un frame y devuelve una lista de ventanas completas si ya hay suficientes.
        Cada ventana es una lista de frames (longitud = window_size).
        """
        self.buffer.append(frame)
        self.counter += 1
        windows = []

        if len(self.buffer) == self.window_size and (self.counter - self.last_emit) >= self.stride:
            windows.append(list(self.buffer))
            self.last_emit = self.counter

        return windows
