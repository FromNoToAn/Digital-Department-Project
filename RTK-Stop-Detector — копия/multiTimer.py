import time

class MultiTimer:
    def __init__(self, num_timers):
        self.timers = {i: {'start_time': 0, 'elapsed_time': 0, 'running': False} for i in range(num_timers)}

    def start_timer(self, timer_id):
        if timer_id not in self.timers:
            raise ValueError("Неверный идентификатор таймера.")
        if not self.timers[timer_id]['running']:
            self.timers[timer_id]['start_time'] = time.time() - self.timers[timer_id]['elapsed_time']
            self.timers[timer_id]['running'] = True

    def pause_timer(self, timer_id):
        if timer_id not in self.timers:
            raise ValueError("Неверный идентификатор таймера.")
        if self.timers[timer_id]['running']:
            self.timers[timer_id]['elapsed_time'] = time.time() - self.timers[timer_id]['start_time']
            self.timers[timer_id]['running'] = False

    def awake_timer(self, timer_id):
        if timer_id not in self.timers:
            raise ValueError("Неверный идентификатор таймера.")
        if not self.timers[timer_id]['running']:
            self.timers[timer_id]['start_time'] = time.time() - self.timers[timer_id]['elapsed_time']
            self.timers[timer_id]['running'] = True
        else:
            raise RuntimeError("Таймер уже запущен!")

    def reset_timer(self, timer_id):
        if timer_id not in self.timers:
            raise ValueError("Неверный идентификатор таймера.")
        self.timers[timer_id] = {'start_time': 0, 'elapsed_time': 0, 'running': False}

    def get_timer(self, timer_id):
        if timer_id not in self.timers:
            raise ValueError("Неверный идентификатор таймера.")
        if self.timers[timer_id]['running']:
            return time.time() - self.timers[timer_id]['start_time']
        return self.timers[timer_id]['elapsed_time']

    def format_time(self, total_seconds):
        hours, remainder = divmod(int(total_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def get_formatted_time(self, timer_id):
        return self.format_time(self.get_timer(timer_id))

# Пример использования
if __name__ == "__main__":
    multi_timer = MultiTimer(3)  # Создаем мультитаймер на 3 таймера

    # Используем первый таймер
    multi_timer.start_timer(0)
    time.sleep(2)  # Ждем 2 секунды
    print("Таймер 0:", multi_timer.get_timer(0))  # Должно вывести 00:00:02
    multi_timer.pause_timer(0)
    time.sleep(1)  # Ждем 1 секунду (таймер не работает)
    print("Таймер 0 после паузы:", multi_timer.get_timer(0))  # Должно вывести 00:00:02
    multi_timer.awake_timer(0)
    time.sleep(1)  # Ждем 1 секунду (таймер снова работает)
    print("Таймер 0:", multi_timer.get_timer(0))  # Должно вывести 00:00:03
    multi_timer.reset_timer(0)
    print("Таймер 0 после сброса:", multi_timer.get_timer(0))  # Должно вывести 00:00:00

    # Используем второй таймер
    multi_timer.start_timer(1)
    time.sleep(1)  # Ждем 1 секунду
    print("Таймер 1:", multi_timer.get_timer(1))  # Должно вывести 00:00:01
    multi_timer.pause_timer(1)
