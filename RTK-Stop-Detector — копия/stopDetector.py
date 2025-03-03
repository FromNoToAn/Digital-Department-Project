import os
import cv2
import numpy as np
import json
from json_Create import JsonFileProcessor
from collections import defaultdict, deque
import time
from Timer import Timer
import argparse

from base_detector import BaseDetector
from utils_V3 import (
    get_sector, draw_grid, draw_count_in_sectors,
    draw_bounding_box, draw_direction_vector, process_Stop_objects,
    update_sector_speed_data, clear_directory, calculate_time_total)

# Список целевых классов объектов для распознавания в модели YOLO (на основе COCO-датасета):
# 2 - Автомобиль (Car), 3 - Мотоцикл (Motorcycle), 5 - Автобус (Bus), 7 - Грузовик (Truck)
TARGET_CLASSES = [2, 3, 5, 7]
COLOR_RED = (0, 0, 255)       # Красный цвет для выделения объектов
COLOR_GREEN = (0, 255, 0)     # Зеленый цвет для выделения объектов
COLOR_YELLOW = (0, 255, 255)  # Желтый цвет для выделения объектов
COLOR_BLUE = (255, 0, 0)      # Синий цвет для выделения области
COLOR_WHITE = (255, 255, 255) # Белый цвет для тестирования
timer = Timer()

class StopDetector:
    def __init__(self, model_path="./model/yolov8s.onnx"):
        self.model = BaseDetector(model_path) # используемая нейронная сеть
        self.timers = {i: Timer() for i in ["start_time"]} # словарь таймеров
        self.track_history = defaultdict(lambda: deque(maxlen=100)) # словарь изменения трэка объектов
        self.sector_time_history = defaultdict(lambda: defaultdict(float)) # словарь времени нахождения объектов в определенном секторе
        self.green_bboxes = defaultdict(lambda: defaultdict(bool)) # bbox зеленых объектов
        self.yellow_bboxes = defaultdict(lambda: defaultdict(bool)) # bbox желтых объектов
        self.red_bboxes = defaultdict(lambda: defaultdict(bool))  # bbox красных объектов
        self.car_stop_time =  defaultdict(lambda:  defaultdict(int)) # считает время пока объект отмечен желтым или красным цветом
        self.color_stop_time = defaultdict(lambda:  defaultdict(int)) # сколько по времени машина стоит
        self.colorChange =  defaultdict(lambda:  defaultdict(int) ) # словарь для отслеживания обеъктов, которые поменяли цвет
        self.colorChangeBool =  defaultdict(lambda:  defaultdict(bool) ) # словарь отслеживающий изменение цвета
        self.red_dir = clear_directory('./Results/Park') # папка для хранения информации об объектах с большим временем стоянки
        self.yellow_dir = clear_directory('./Results/Stop') # папка для хранения информации об объектах с коротким временем стоянки
        self.time_stop = 0 # время остановки объекта с выбранным id
        self.drawing_mode = False # режим разметки (True/да, False/нет)
        self.region_count=10, # число зон разметки по умолчанию 10
        self.grid_size = (24, 24) # размер сетки 
        self.video_record = True # хаписываем ли видео
        self.frame_size  = (1280, 720) # разрешение видео
        self.frame_skip=True # пропуск кадров для ускорения обработки
        self.alarm_time = 1.5 # время необходимое для покраски в желтый цвет
        self.timePark = 5 # время в секундах с которого записываем в папку Park
        self.output_interval = 5 # интервал записи файлов и их фильтрации в папку
        self.TimeFolder_stop = 5 # время с которого записываем кропы в папку Stop
        self.JsonWrite_stop=True # сохраняем ли json в папку Stop
        self.CropWrite_stop=False # сохраняем ли crops в папку Stop
        self.TimeFolder_park = 5 # время с которого записываем кропы в папку Stop
        self.JsonWrite_park=True # сохраняем ли json в папку Stop
        self.CropWrite_park=False # сохраняем ли crops в папку Stop
        self.frame_to_avg = 3 # число кадров для усреднения
        self.MaxLen=1000 # максимальная длина массивов
        self.time_Total=0 # инициализация переменной
        self.count_reset = 1000 # сброс счетчика в секторах
        self.sector_size = (self.frame_size[0] // self.grid_size[0], self.frame_size[1] // self.grid_size[1])
        self.sector_speed_data = np.zeros(
            (self.grid_size[1], self.grid_size[0]),
            dtype=[('count', 'i4'), ('avg_vector', 'f4')]
        ) # Инициализация сектора скорости
        self.polygon_points = [] # Инициализация областей для разметки
        self.color = COLOR_WHITE
        # Инициализация цветов для секторов
        self.COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 64, 64)
        ]
        self.stop_flag = False  # Флаг остановки обработки видео

    def get_object_color(self, track_id, alarm_time, timePark, elapsed_time, norm_vector_length, current_sector, MaxLen):
        # print("id=", track_id, self.time_red)
        self.green_bboxes=self.update_dict_maxlen(self.green_bboxes, MaxLen) # ограничиваем длину словаря
        self.yellow_bboxes=self.update_dict_maxlen(self.yellow_bboxes, MaxLen) # ограничиваем длину словаря
        self.red_bboxes=self.update_dict_maxlen(self.red_bboxes, MaxLen) # ограничиваем длину словаря
        # print(len(self.green_bboxes),len(self.yellow_bboxes),len(self.red_bboxes) )
        
        # Если объект стоит больше self.time_stop он окрасится в красный цвет
        if self.time_stop > timePark:
            self.red_bboxes[track_id][current_sector] = True
            return COLOR_RED
        
        # Если время нахождения объекта в секторе больше чем elapsed_time, то объект окрасится в желтый цвет
        if ((elapsed_time  > alarm_time)
              or self.yellow_bboxes[current_sector][track_id]
              or norm_vector_length < 0.02):
            self.yellow_bboxes[current_sector][track_id] = True
            # if self.time_stop > timePark and self.yellow_bboxes[current_sector][track_id]:
            #     self.red_bboxes[track_id][current_sector] = True
            #     return COLOR_RED
            self.green_bboxes[current_sector][track_id]=False
            return COLOR_YELLOW
        
        # Если объект поменял свой сектор, то он вновь окрасится в зеленый цвет
        elif self.colorChangeBool[track_id][current_sector]:
            self.green_bboxes[current_sector][track_id]=True
            self.yellow_bboxes[current_sector][track_id]=False
            self.red_bboxes[current_sector][track_id]=False
            # if (self.colorChangeBool[track_id][current_sector]):print("id=",track_id, "Change to GREEN")
            return COLOR_GREEN
        
        # Иначе объект окрасится в зеленый цвет
        else:
            self.green_bboxes[current_sector][track_id]=True
            self.yellow_bboxes[current_sector][track_id]=False
            self.red_bboxes[current_sector][track_id]=False
            return COLOR_GREEN

    # функция для ограничения длины словарей
    def update_dict_maxlen(self, dictionary, MaxLen):
        """Обновляет любой словарь с ограничением длины записей"""
        if len(dictionary) > MaxLen:
            oldest_key = next(iter(dictionary))
            del dictionary[oldest_key]
        return dictionary

    # Обрабатывает данные об остановках автомобиля по секторам
    def check_alarm(self, car_stop_data):
        """
        Обрабатывает данные об остановках автомобиля по секторам.
        Возвращает словарь с временем остановки в каждом уникальном секторе.

        :param car_stop_data: словарь, где ключи — это секторы, 
                            а значения — время остановки.
        :return: словарь с уникальными секторами и их временем остановки.
        """
        # Проверяем, что входные данные car_stop_data не пусты. Если пусто, возвращаем пустой словарь.
        if not car_stop_data:
            return {}
        current_time = {}  # Инициализируем словарь для хранения времени для каждого сектора
        previous_sector = None
        for current_sector in car_stop_data:
            # Если текущий сектор не равен предыдущему, добавляем данные в current_time
            if previous_sector is None or current_sector != previous_sector:
                current_time[current_sector] = car_stop_data[current_sector]
            previous_sector = current_sector
        # print(list(car_stop_data.keys()), current_time)
        return current_time  # Возвращаем итоговый результат

    # Обрабатывает данные о срабатывании тревог (ALARM) на основе заданного порога
    def check_alarm_and_conditions(self, dict_alrm, threshold):
        """
        Обрабатывает данные о срабатывании тревог (ALARM) на основе заданного порога
        и выполняет дополнительные проверки соседних секторов. Возвращает количество
        срабатываний ALARM.
        :param dict_alrm: словарь с данными по секторам, где ключи — сектора, 
                        а значения — времена срабатывания.
        :param threshold: пороговое значение времени, выше которого считается ALARM.
        :return: количество срабатываний ALARM.
        """
        # Список для хранения разделенных данных.
        split_dicts = []
        # Текущий словарь для сбора данных по секторам.
        current_dict = {}
        # Счетчик срабатываний ALARM.
        alarm_count = 0
        # Список для хранения промежуточных результатов (опционально, не используется).
        list_count = []
        # Перебираем данные по секторам.
        for sector, time_sector in dict_alrm.items():
            # Если текущее время превышает порог и текущий словарь не пуст,
            # добавляем его в список и очищаем для нового блока данных.
            if time_sector > threshold and current_dict:
                split_dicts.append({'data': current_dict})  # Сохраняем текущий словарь.
                current_dict = {}  # Очищаем для нового блока.
            # Добавляем данные текущего сектора в словарь.
            current_dict[sector] = time_sector
        # После завершения итерации добавляем последний собранный словарь, если он не пуст.
        if current_dict:
            split_dicts.append({'data': current_dict})
        # Проверка условий для каждого разделенного блока данных.
        for split_dict in split_dicts:
            dict_alrm = split_dict['data']
            # Проверяем, есть ли хотя бы одно значение, превышающее порог,
            # и одновременно существуют ли соседние сектора с меньшими значениями.
            if any(t > threshold for t in dict_alrm.values()) and any(
                (x + dx, y + dy) in dict_alrm and dict_alrm[(x + dx, y + dy)] < threshold
                for (x, y), t in dict_alrm.items() if t < threshold
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            ):
                # Если условие выполняется, увеличиваем счетчик срабатываний ALARM.
                alarm_count += 1
                # Очищаем данные текущего словаря после обработки.
                split_dict['data'] = {}
        # Возвращаем общее количество срабатываний ALARM.
        return alarm_count

    # метод для очистки данных
    def clear_data(self):
        self.timers = {i: Timer() for i in ["start_time"]} # словарь таймеров
        self.track_history = {} 
        self.track_history = defaultdict(lambda: deque(maxlen=100)) # словарь изменения трэка объектов
        self.sector_time_history = {} 
        self.sector_time_history = defaultdict(lambda: defaultdict(float)) # словарь времени нахождения объектов в определенном секторе
        self.green_bboxes = {}
        self.green_bboxes = defaultdict(lambda: defaultdict(bool)) # bbox зеленых объектов
        self.yellow_bboxes = {}
        self.yellow_bboxes = defaultdict(lambda: defaultdict(bool)) # bbox желтых объектов
        self.red_bboxes = {}
        self.red_bboxes = defaultdict(lambda: defaultdict(bool))  # bbox красных объектов
        self.car_stop_time = {}
        self.car_stop_time =  defaultdict(lambda:  defaultdict(int)) # считает время пока объект отмечен желтым или красным цветом
        self.color_stop_time = {}
        self.color_stop_time = defaultdict(lambda:  defaultdict(int)) # сколько по времени машина стоит
        self.colorChange = {}
        self.colorChange =  defaultdict(lambda:  defaultdict(int) ) # словарь для отслеживания обеъктов, которые поменяли цвет
        self.colorChangeBool = {}
        self.colorChangeBool =  defaultdict(lambda:  defaultdict(bool) ) # словарь отслеживающий изменение цвета

    # основной метод обработки
    def run(self, video_path = None, time_Total=0,
            drawing_mode = 0, # режим разметки (True/да, False/нет)
            region_count=10, # число зон разметки по умолчанию 10
            grid_size = (32, 32), # размер сетки 
            video_record = True, # записываем файл видео с результатом (True/да, False/нет)
            frame_size  = (1280, 720), # разрешение видео
            frame_skip=True, # пропуск кадров для ускорения обработки
            alarm_time = 1.5, # время необходимое для покраски в желтый цвет
            output_interval = 5, # интервал записи файлов и их фильтрации в папку
            TimeFolder_stop = 5, # время с которого записываем кропы в папку Stop
            JsonWrite_stop=False, # сохраняем ли json в папку Stop
            CropWrite_stop=False, # сохраняем ли crops в папку Stop
            TimeFolder_park = 10, # время с которого записываем кропы в папку Park
            JsonWrite_park=True, # сохраняем ли json в папку Stop
            CropWrite_park=True, # сохраняем ли crops в папку Stop
            frame_to_avg = 3, # число кадров для усреднения
            MaxLen=1000, # максимальная длина словарей
            count_reset = 1000 # сброс счетчика в секторах
            ):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.stop_flag = False  # Теперь обращаемся к self.stop_flag # Флаг остановки всей программы
        sector_size = (frame_size[0] // grid_size[0], frame_size[1] // grid_size[1])
        current_time = time.strftime("%Y-%m-%d_%H-%M-%S") # Текущее время в формате "ГГГГ-ММ-ДД_ЧЧ-ММ-СС"
        file_name = f'./Results/{current_time}.mp4'  # Сохранение видео в папке Results
        if video_record:
            video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 25, frame_size)

        # Инициализация сектора скорости
        sector_speed_data = np.zeros(
            (grid_size[1], grid_size[0]),
            dtype=[('count', 'i4'), ('avg_vector', 'f4')]
        )
        #Обработка регионов
        polygons = []  # Список для хранения координат всех полигонов
        time_park_dict = {}  # Словарь для хранения времени парковки для каждого полигона 
        region_names = {}  # Словарь для хранения названий регионов 
        region_centers = {}  # Словарь для хранения центров регионов      
        # Если drawing_mode==True, тогда создаем зоны и json файл
        if drawing_mode:
            processor = JsonFileProcessor(video_path, regions_file = 'regions.json', 
                                            frame_size=frame_size, makeJson=1, region_count=region_count)
            processor.process()
        
        # Загрузка данных из JSON-файла
        with open('regions.json', 'r') as f:
            polygon_data = json.load(f)
            # Извлечение точек полигона
            
            # Заполняем полигоны и время парковки
        for i, shape in enumerate(polygon_data['shapes']):
            points = np.array(shape['points'], dtype=np.int32)
            polygons.append(points)
            time_park_dict[i] = shape.get('flags')  # Используем поле timePark из JSON или значение по умолчанию
            region_names[i] = shape.get('label')  # Используем label или имя по умолчанию

        # print(time_park_dict)    
        # print("Названия регионов:", region_names)

        self.timers["start_time"].start_timer()
        yellow_object_ids = [] # Список для хранения ID объектов с жёлтым цветом
        red_object_ids = []  # Список для хранения ID объектов с красным цветом

        # Основной цикл обработки видео
        actual_frame_duration = (1 / fps) if fps>0 else (1/25)  # Время на один кадр
        frame_skip_timer = time.monotonic() # отслеживания времени между кадрами и контроля частоты обработки с частотой actual_fps
        processed_frames = 0  # Счетчик обработанных кадров
        # start_frame_time = time.monotonic()  # Время начала обработки
        start_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        color = None
        while cap.isOpened():
            timer.restart_timer()
            # start_time = time.monotonic()
            i = 0
            frame_time = time.monotonic() - frame_skip_timer
            if frame_skip:#FRAME SKIP
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # print(f"{time.monotonic()} - {frame_skip_timer} = {time.monotonic() - frame_skip_timer} ({time.monotonic() - frame_skip_timer < actual_frame_duration})")
                    # Вычисляем, сколько времени прошло с момента начала обработки
                    if frame_time < actual_frame_duration or i == 15:
                        # print(f"Кадр актуален. {round(time.monotonic() - frame_skip_timer,3)}")
                        break
                    else:
                        i += 1
                        frame_time -= 1 / fps
                        # print(frame_time, f"Пропущен кадр для актуализации. {time.monotonic()} - {frame_skip_timer} = {round(time.monotonic() - frame_skip_timer,3)}")
                frame_skip_timer = time.monotonic()
                processed_frames += 1  # Увеличиваем счетчик обработанных кадров
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                # frame_time = time.monotonic() - start_frame_time
                frame_time = timestamp - start_frame_time
                real_fps = processed_frames / frame_time if frame_time > 0 else 1  # Избегаем деления на ноль
                # print(f"timestamp={np.round(timestamp, 2)}, frame_time={np.round(frame_time, 2)}, real_fps={np.round(real_fps, 2)}")

            else:#FRAME SKIP
                ret, frame = cap.read()
                if not ret:
                    break
                # Считываем время текущего кадра
                frame_noskip_timer = time.monotonic()  # Получаем время в секундах с начала работы программы
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frame_time = frame_noskip_timer - frame_skip_timer   # Время между кадрами
                real_fps = 1 / frame_time if frame_time > 0 else 0  # FPS обработки
                # print(f"timestamp={np.round(timestamp, 2)}, real_fps={np.round(real_fps, 2)}")
                frame_skip_timer = frame_noskip_timer
            if frame is None:#FRAME SKIP
                break#FRAME SKIP

            frame = cv2.resize(frame, frame_size)
            results = self.model.run(frame)
            yellow_object_ids = [] # Список для хранения ID объектов с жёлтым цветом
            red_object_ids = []  # Список для хранения ID объектов с красным цветом
            # Проверка и замыкание полигона
            
            displayed_regions = [] # список выводимых названий полигонов
            # Если drawing_mode=0 тогда не рисуем зоны, но если они есть в файле выводим их
            if not drawing_mode or len(region_names)!=0:
                for i, polygon in enumerate(polygons):
                    # Если текст для этого региона ещё не был выведен
                    if i not in displayed_regions:
                        # cv2.polylines(frame, [polygon], isClosed=True, color=(255, 0, 0), thickness=2) # Рисуем полигон
                        cv2.polylines(frame, [polygon], isClosed=True, color=self.COLORS[i % len(self.COLORS)] , thickness=2)  # Рисуем полигон
                          
                        region_name = region_names.get(i, "Unknown") 
                        time_park = time_park_dict.get(i, "N/A")
                        display_text = f"{region_name}_t={time_park}s" # Формируем текст для вывода
                        # Рассчитываем центр полигона
                        moments = cv2.moments(polygon)
                        # if moments["m00"] != 0:
                        #     cX = int(moments["m10"] / moments["m00"]) 
                        #     cY = int(moments["m01"] / moments["m00"])
                        # else:
                        cX, cY = polygon[0][0], polygon[0][1] # или используем первую точку полигона
                        # Выводим название региона только один раз для каждого полигона
                        cv2.putText(frame, display_text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                        displayed_regions.append(i) # Добавляем регион в список выведенных

            if results is not None:
                try:
                    boxes = [result[0] for result in results]
                    track_ids = [result[1] for result in results]
                    classes = [result[2] for result in results]
                    confidences = [result[3] for result in results]
                except AttributeError as e:
                    print(f"Error: {e}")
                    continue

                annotated_frame = frame.copy()
                # draw_grid(annotated_frame, sector_size, grid_size, COLOR_WHITE)
                # draw_count_in_sectors(annotated_frame, sector_speed_data, sector_size, grid_size, COLOR_WHITE)

                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    if cls in TARGET_CLASSES:
                        x1, y1, x2, y2 = box.astype(np.int32)
                        w, h = abs(x2 - x1), abs(y2 - y1)
                        x, y = x1 + w // 2, y1 + h // 2

                        # Определяем, в каком полигоне находится объект
                        current_polygon_index = None
                        for i, polygon in enumerate(polygons):
                            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                                current_polygon_index = i
                                break                       
                       
                        if current_polygon_index is not None:
                            time_park = time_park_dict[current_polygon_index]  # для каждой области определяем время парковки                   
                            track = self.track_history[track_id] # print(len(track))
                            track.append((float(x), float(y)))  # координаты центра

                            result = draw_direction_vector(annotated_frame, track, frame_to_avg, COLOR_YELLOW, (x, y, w, h))
                            norm_vector_length, _ = result if result is not None else (0, 0)  # Значения по умолчанию


                            if len(track) > 1:
                                # previous_sector = get_sector(int(track[-2][0]), int(track[-2][1]), sector_size, grid_size)
                                current_sector = get_sector(int(x), int(y), sector_size, grid_size)
                                update_sector_speed_data(current_sector, norm_vector_length, sector_speed_data, count_reset)

                                actual_elapsed_time = (1 / real_fps) if frame_skip else (1 / fps)
                                self.sector_time_history[current_sector][track_id] += actual_elapsed_time
                                self.sector_time_history=self.update_dict_maxlen(self.sector_time_history, MaxLen) # ограничиваем длину словаря
                                elapsed_time = np.round(self.sector_time_history[current_sector][track_id], 2)
                                # print("track_id=",track_id, "elapsed_time", elapsed_time)

                                if elapsed_time>alarm_time and (color == COLOR_YELLOW or color == COLOR_RED):
                                    self.car_stop_time[track_id][current_sector] = elapsed_time
                                    self.car_stop_time=self.update_dict_maxlen(self.car_stop_time, MaxLen) # ограничиваем длину словаря
                                    # print(len(self.car_stop_time))
                                # полное время остановки/стоянки объекта
                                if self.car_stop_time[track_id]:
                                    time_Total = np.round(calculate_time_total(self.car_stop_time[track_id]),2) # время только остановки/стоянки
                                    # print("Total time=", time_Total, self.car_stop_time[track_id])

                                # сколько по времени машина стоит этот блок нужен для отслеживания смены сектора машины с red bbox
                                if color == COLOR_YELLOW or color == COLOR_GREEN or color == COLOR_RED:
                                    self.color_stop_time[track_id][current_sector] = elapsed_time
                                    self.color_stop_time=self.update_dict_maxlen(self.color_stop_time, MaxLen) # ограничиваем длину словаря
                                    self.time_stop=np.round(calculate_time_total(self.color_stop_time[track_id]),2)
                                    # print(self.car_stop1_time[track_id])
                                    # print("Y_time=", elapsed_time,"id=", track_id, self.car_stop1_time[track_id])

                                dict_alarm = self.check_alarm(self.color_stop_time[track_id]) # словарь машин, которые меняют цвет
                                dict_alarm=self.update_dict_maxlen(dict_alarm, MaxLen) # ограничиваем длину словаря
                                # print(len(dict_alarm))

                                count_Y_toGr = self.check_alarm_and_conditions(dict_alarm, threshold=alarm_time)
                                if count_Y_toGr not in self.colorChange[track_id].values() and count_Y_toGr!=0:
                                    # print(count_Y_toGr)
                                    self.colorChange[track_id][current_sector] = count_Y_toGr
                                    self.colorChange=self.update_dict_maxlen(self.colorChange, MaxLen) # ограничили длину словаря
                                    self.colorChangeBool[track_id][current_sector]=True
                                    self.colorChangeBool=self.update_dict_maxlen(self.colorChangeBool, MaxLen) # ограничили длину словаря
                                    self.color_stop_time.clear()
                                    self.time_stop=0

                                # Рассчитываем площадь bbox и сектора
                                bbox_area = w * h
                                sector_area = sector_size[0] * sector_size[1]
                                if (grid_size[0] or grid_size[1])==16:
                                    if bbox_area < 1.2 * sector_area:
                                            continue  # Пропускаем текущий объект
                                if (grid_size[0] or grid_size[1])>16:
                                    if bbox_area < 5.0 * sector_area:
                                            continue  # Пропускаем текущий объект

                                # Логика окрашивания объектов
                                color = self.get_object_color(track_id, alarm_time, time_park, elapsed_time,
                                                              norm_vector_length, current_sector, MaxLen)
                                # Отрисовка bbox и информации
                                draw_bounding_box(annotated_frame, track_id, self.time_stop,
                                                  norm_vector_length, (x, y, w, h), color)
                                if color == COLOR_YELLOW: # and time_since_reset>0.01
                                    yellow_object_ids.append((track_id, time_Total, current_polygon_index+1, np.round(frame_time,2), processed_frames))
                                    yellow_object_ids = yellow_object_ids[-MaxLen:] # ограничиваем список вывода
                                if color == COLOR_RED:
                                    red_object_ids.append((track_id, time_Total, current_polygon_index+1, np.round(frame_time,2), processed_frames))
                
                # if (red_object_ids):
                #     print(red_object_ids)

                # если поток с видеокадрами запущен, тогда записываем информацию
                if real_fps > 0:
                    if int(self.timers["start_time"].get_time()) >= output_interval:
                        self.timers["start_time"].restart_timer()
                        process_Stop_objects(yellow_object_ids, boxes, track_ids,
                                             frame, self.yellow_dir, TimeFolder_stop , JsonWrite_stop, CropWrite_stop)
                        process_Stop_objects(red_object_ids, boxes, track_ids,
                                             frame, self.red_dir, TimeFolder_park , JsonWrite_park, CropWrite_park)
                        # stop_images = filter_similarity(self.yellow_dir, timeStop) # фильтрация IFT
                        # print(filtered_images)

                # Выводим информацию на экран
                cv2.putText(annotated_frame, f"FPS: {real_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
                if video_record:
                    video_writer.write(annotated_frame)
                cv2.imshow("lane_stop_detector", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.stop_flag = True  # Устанавливаем флаг остановки  # Устанавливаем флаг остановки
                break

        video_writer.release()
        cap.release()
        cv2.destroyAllWindows()
        # return [yellow_object_ids, red_object_ids]
        return

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Запуск детектора остановки")
#     parser.add_argument('--video_path', type=str, help="Путь к видеофайлу/RTSP-потоку", default=None)
#     parser.add_argument('--video_directory', type=str, help="Путь к каталогу с видеофайлами", default=None)
#     parser.add_argument('--rtsp_url', type=str, help="RTSP URL", default=None)
#     parser.add_argument('--drawing_mode', type=int, help="Разметка данных Да/Нет (1/0)", default=0)
#     args = parser.parse_args()

#     stopDetector = StopDetector('./models/yolov8s_mod.onnx')

#     # Определяем, что использовать: video_path или video_directory
#     if args.video_path:
#         # Обработка одного видео
#         video_path = args.video_path
#         print(f"Обработка видео: {video_path}")
#         result = stopDetector.run(video_path, drawing_mode=bool(args.drawing_mode))
#         if result is None:
#             print("Видео остановлено.")
#         else:
#             print(result)
#     elif args.video_directory:
#         # Обработка всех видео в каталоге
#         video_files = [f for f in os.listdir(args.video_directory) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
#         if not video_files:
#             print("В каталоге нет видеофайлов.")
#         else:
#             for video_file in video_files:
#                 print(f"Обработка видео: {video_file}")
#                 video_path = os.path.join(args.video_directory, video_file)
#                 result = stopDetector.run(video_path, drawing_mode=bool(args.drawing_mode))
#                 if result is None:
#                     print(f"Видео {video_file} остановлено.")
#                 else:
#                     print(result)
#     elif args.rtsp_url:
#         # Обработка RTSP потока
#         video_path = args.rtsp_url
#         print(f"Обработка RTSP потока: {video_path}")
#         result = stopDetector.run(video_path, drawing_mode=bool(args.drawing_mode))
#         if result is None:
#             print("RTSP поток остановлен.")
#         else:
#             print(result)
#     else:
#         print("Не задан путь к видео, каталогу или RTSP потоку.")

def process_video(source, detector, drawing_mode, source_type="видео"):
    print(f"Обработка {source_type}: {source}")
    result = detector.run(source, drawing_mode=drawing_mode)
    status = "остановлено." if result is None else result
    print(f"Результат ({source}): {status}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Запуск детектора остановки")
    parser.add_argument('--video_path', help="Путь к видеофайлу/RTSP-потоку")
    parser.add_argument('--video_directory', help="Путь к каталогу с видеофайлами")
    parser.add_argument('--rtsp_url', help="RTSP URL")
    parser.add_argument('--drawing_mode', type=int, choices=[0,1], default=0, 
                      help="Разметка данных Да/Нет (1/0)")
    
    args = parser.parse_args()
    detector = StopDetector('./models/yolov8s_mod.onnx')
    drawing_mode = bool(args.drawing_mode)

    # Определение приоритета источников
    if args.video_path:
        process_video(args.video_path, detector, drawing_mode)
    elif args.video_directory:
        video_files = [f for f in os.listdir(args.video_directory) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for f in video_files or ["Нет файлов"]:
            if f == "Нет файлов": 
                print("В каталоге нет видеофайлов.")
                break
            process_video(os.path.join(args.video_directory, f), detector, drawing_mode)
            if detector.stop_flag:  # Если флаг установлен, полностью выходим
                break
    elif args.rtsp_url:
        process_video(args.rtsp_url, detector, drawing_mode, "RTSP потока")
    else:
        print("Не задан ни один источник для обработки.")

# Run detector       
# python stopDetector.py --video_path "./Videos/output_070.mp4" --drawing_mode=0
# python stopDetector.py --video_directory "./Videos" 
# python stopDetector.py --rtsp_url "rtsp://admin:lavrentiev6@10.2.21.221/cam/realmonitor?channel=1&subtype=1" --drawing_mode 1
