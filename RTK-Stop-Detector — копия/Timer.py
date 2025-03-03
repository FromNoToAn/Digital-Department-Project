import functools
import threading
import time
import numpy as np

class Timer:
    def __init__(self, thread = False, tick = 1):
        self.start_time = 0
        self.elapsed_time = 0
        self.running = False
        self.pause_history = [] 
        self._tick = tick
        self._main_thread = None
        self._timer_thread = None
        self.current_time = 0
        self.notify = False
        self._duration = 0
        self._on_timer = False
        self._timer_last_time = 0
        self._thread = thread
    
    @staticmethod
    def time_logger(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.monotonic()
            result = func(*args, **kwargs)
            end_time = time.monotonic()
            execution_time = end_time - start_time
            print(f"Time of '{func.__name__}': {execution_time:.3f} seconds")
            return result
        return wrapper

    def _thread_create(self, func):
        _thread = threading.Thread(target=func)
        _thread.daemon = True
        return _thread
                
    def _ticking(self):
        #while True:
            while self.running:
                time.sleep(self._tick)
                self.current_time = self.get_time()
                if self._on_timer:
                    if self.current_time >= self.current_time + self.duration:
                        self._timer_last_time = self.current_time
                        self.notify = True
                        self._on_timer = False
                    else:
                        self.duration -= self._tick
    
    def _timer_count(self):
        while self._on_timer:
            time.sleep(self._tick)
            print("duration", self.duration)
            print(f"{self.current_time} >= {self.current_time + self.duration}")
            if self.current_time >= self.current_time + self.duration:
                
                self._timer_last_time = self.current_time
                self.notify = True
                self._on_timer = False
            else:
                self.duration -= self._tick
            
    def start_timer(self):
        if not self.running:
            self.start_time = time.monotonic() - self.elapsed_time
            self.running = True
            if self._thread:
                self._main_thread = self._thread_create(self._ticking)
                self._main_thread.start()

    def restart_timer(self):
        self.reset_timer()
        self.start_timer()
        
    def pause_timer(self):
        if self.running:
            self.elapsed_time = self.get_time()
            self.pause_history.append(self.elapsed_time)
            self.running = False
            if self._thread:
                self._main_thread.join()
                self._main_thread = self._thread_create(self._ticking)

    def awake_timer(self):
        if not self.running:
            self.start_time = time.monotonic() - self.elapsed_time
            self.running = True
            if self._thread:
                self._main_thread.start()
        else:
            raise RuntimeError("Таймер уже запущен!")

    def reset_timer(self):
        self.start_time = 0
        self.elapsed_time = 0
        self.running = False
        if self._thread:
            self._main_thread.join()
        self.current_time = 0
        self.pause_history.clear()
        

    def get_time(self):
        if self.running:
            return time.monotonic() - self.start_time
        return self.elapsed_time

    def get_start_time(self):
        return self.start_time
    
    def format_time(self, total_seconds):
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def get_formatted_time(self):
        return self.format_time(self.get_timer())

    def stop_timer(self):
        self.pause_timer()
        self.reset_timer()

    def is_running(self):
        return self.running

    def set_timer(self,duration=0):
        if not self._on_timer: 
            self.duration = duration
            self.notify = False
            self._on_timer = True
            self._timer_thread = self._thread_create(self._timer_count)
            self._timer_thread.start()

    def get_timer_done(self):
        return self.notify 
    
    def get_timer_time(self):
        return self._timer_last_time

    def get_timer_history(self):
        return self.pause_history

class MultiTimer(Timer):
    def __init__(self, num_timers):
        self.timers = {i: {'start_time': 0, 'elapsed_time': 0, 'running': False} for i in range(num_timers)}    
# Пример использования        
if __name__ == "__main__":
    timer = Timer()
    timer.start_timer()
    for i in range(69999):
        a = i ** 2
        print("current time on: ", timer.current_time)
    timer.pause_timer()
    for i in range(39999):
        a = i ** 2
        print("current time off: ", timer.current_time)
    timer.awake_timer()
    for i in range(59999):
        a = i ** 2
        print("current time on: ", timer.current_time)
    