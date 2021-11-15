"""

Profiler.

"""

# Copyright (c) 2021 Ben Zimmer. All rights reserved.

from collections import OrderedDict
import time


class Profiler:
    """simple class for totalling up times"""

    def __init__(self):
        """create a collection of timers"""
        self.times = OrderedDict()

    def tick(self, key):
        """start a timer"""
        _, total_time = self.times.setdefault(key, (None, 0.0))
        self.times[key] = (time.time(), total_time)

    def tock(self, key):
        """stop a timer"""
        end_time = time.time()
        start_time, total_time = self.times.setdefault(key, (end_time, 0.0))
        self.times[key] = (None, total_time + (end_time - start_time))

    def summary(self):
        """pretty-print results"""
        times = [(x, y) for x, (_, y) in self.times.items()]
        key_len_max = max(len(x) for x in self.times.keys())
        for key, dur in times:
            print(key.ljust(key_len_max + 4, ".") + str(round(dur, 3)))
