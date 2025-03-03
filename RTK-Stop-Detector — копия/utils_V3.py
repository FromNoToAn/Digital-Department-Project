import cv2
import numpy as np
import json
import time
import os
import shutil 
import re
import glob
from collections import OrderedDict


def clear_directory(directory):
    # Если папка существует, удаляем все файлы в ней
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            # Проверяем, является ли путь файлом, и удаляем его
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(directory)
    return directory

def get_sector(x, y, sector_size, grid_size):
    x_idx = min(x // sector_size[0], grid_size[0] - 1)
    y_idx = min(y // sector_size[1], grid_size[1] - 1)
    return y_idx, x_idx

def draw_grid(frame, sector_size, grid_size, color_white):
    for i in range(1, grid_size[0]):
        cv2.line(frame, (i * sector_size[0], 0), (i * sector_size[0], frame.shape[0]), color_white, 1)
    for j in range(1, grid_size[1]):
        cv2.line(frame, (0, j * sector_size[1]), (frame.shape[1], j * sector_size[1]), color_white, 1)


def process_Stop_objects(object_ids, boxes, track_ids, annotated_frame, current_dir, MaxTime=20, JsonWrite=True, CropWrite=False):
    # Создаем словарь bbox для каждого track_id
    bbox_dict = {tid: box.astype(np.int32) for box, tid in zip(boxes, track_ids)}
    filtered_object_ids = []

    if object_ids:
        for track_id, time_stop, sector_stop, start_time, start_frame in object_ids:
            # Преобразуем start_time в формат 'часы:минуты:секунды'
            hours = int(start_time // 3600)  # Часы
            minutes = int((start_time % 3600) // 60)  # Минуты
            seconds = int(start_time % 60)  # Секунды
            # Формируем строку в формате "часы:минуты:секунды"
            formatted_time = f"{hours:02}:{minutes:02}:{seconds:02}"
            # Фильтруем объекты по времени
            if time_stop > MaxTime and track_id in bbox_dict:
                filtered_object_ids.append((track_id, time_stop, sector_stop))  # Сохраняем отфильтрованные данные
                if CropWrite:  # Сохраняем кропы только если CropWrite=True
                    x1, y1, x2, y2 = bbox_dict[track_id]
                    h, w = annotated_frame.shape[:2]
                    x1, y1, x2, y2 = max(0, min(x1, w)), max(0, min(y1, h)), max(0, min(x2, w)), max(0, min(y2, h))
                    if x2 > x1 and y2 > y1:
                        crop_img = annotated_frame[y1:y2, x1:x2]
                        if crop_img.size > 0:
                            new_file_name = os.path.join(current_dir, f"Reg{sector_stop}-id={track_id}_timeStop={time_stop:.2f}s.jpg")
                            # cv2.imwrite(os.path.join(current_dir, f"Reg{sector_stop}-id={track_id}_timeStop={time_stop:.2f}s.jpg"), crop_img)                    
                            cv2.imwrite(new_file_name, crop_img)
                    # crop_img = annotated_frame[y1:y2, x1:x2]
                    
    # Запись в JSON при необходимости
    if JsonWrite:
        # json_file_path = os.path.join(current_dir, current_dir+".json")
        json_file_path = os.path.join(current_dir, os.path.basename(current_dir) + ".json")

        
        # Загрузка существующего JSON-файла (если он есть)
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as json_file:
                try:
                    existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        # Преобразование списка в словарь для удобства обновления
        for track_id, time_stop, sector_stop in filtered_object_ids:
            track_id_str = str(track_id)  # Преобразуем ID в строку для использования в JSON
            if track_id_str in existing_data:
                if "start_time" not in existing_data[track_id_str]:
                    existing_data[track_id_str]["start_time"] = formatted_time # время начала события
                if "start_frame" not in existing_data[track_id_str]:
                    existing_data[track_id_str]["start_frame"] = start_frame # номер кадра начала события
                existing_data[track_id_str]["time"] = time_stop
                existing_data[track_id_str]["region"] = sector_stop  # Обновляем сектор
            else:
                existing_data[track_id_str] = {
                        "start_time": formatted_time,
                        "start_frame": start_frame,
                        "time": time_stop,
                        "region": sector_stop
                        }
                
        # Сохраняем обновленные данные в JSON-файл
        with open(json_file_path, "w") as json_file:
            json.dump(existing_data, json_file, indent=4)

    # Получаем список всех файлов и находим максимальное время остановки для каждого ID
    files = glob.glob(os.path.join(current_dir, "*.jpg"))
    max_timeStop_files = OrderedDict()  # Словарь для хранения файлов с максимальным timeStop

    for file in files:
        filename = os.path.basename(file)
        try:
            track_id, time_stop_str = filename.split('_')[:2]
            time_stop = float(time_stop_str.split('=')[1].replace('s.jpg', ''))
            
            # Проверяем, нужно ли обновить max_timeStop_files
            if track_id not in max_timeStop_files or time_stop > max_timeStop_files[track_id][1]:
                max_timeStop_files[track_id] = (file, time_stop)

                # Удаляем самый старый элемент, если длина превышает 1000
                if len(max_timeStop_files) > 1000:
                    max_timeStop_files.popitem(last=False)  # Удаляем первый элемент (самый старый)

        except (IndexError, ValueError) as e:
            continue

    # Удаляем файлы, которые не содержат максимальное timeStop для каждого ID
    for file in files:
        if not any(file == max_timeStop_files[track_id][0] for track_id in max_timeStop_files):
            os.remove(file)

    return max_timeStop_files



def calculate_time_total(car_stop_data):
    # Сортируем сектора для последовательной обработки
    sorted_sectors = sorted(car_stop_data.keys())
    processed_data = {}
    i = 0
    while i < len(sorted_sectors):
        current_sector = sorted_sectors[i]
        max_time = car_stop_data[current_sector]  
        # Ищем все соседние сектора и берем максимальное время из группы
        j = i + 1
        while j < len(sorted_sectors) and abs(current_sector[0] - sorted_sectors[j][0]) <= 1 and abs(current_sector[1] - sorted_sectors[j][1]) <= 1:
            max_time = max(max_time, car_stop_data[sorted_sectors[j]])
            j += 1
        # Сохраняем максимальное время для группы соседних секторов
        processed_data[current_sector] = max_time
        i = j  # Переходим к следующему сектору после группы
    # Суммируем времена для несоседних секторов
    time_total = sum(processed_data.values())
    return time_total


def draw_avg_speed_in_sectors(frame, sector_speed_data, sector_size, grid_size, color_white):
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX 
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            sector = (y, x)
            avg_speed = sector_speed_data[sector]['avg_speed']
            label = f"{avg_speed:.2f}"
            sector_center_x = int((x + 0.5) * sector_size[0])
            sector_center_y = int((y + 0.5) * sector_size[1])
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = int(sector_center_x - text_size[0] / 2)
            text_y = int(sector_center_y + text_size[1] / 2)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, color_white, thickness)
            
def draw_count_in_sectors(frame, sector_speed_data, sector_size, grid_size, color_white):
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            sector = (y, x)
            avg_speed = sector_speed_data[sector]['count']
            label = f"{avg_speed:.2f}"
            sector_center_x = int((x + 0.5) * sector_size[0])
            sector_center_y = int((y + 0.5) * sector_size[1])
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = int(sector_center_x - text_size[0] / 2)
            text_y = int(sector_center_y + text_size[1] / 2)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, color_white, thickness)            

def draw_avg_vectors_in_sectors(frame, sector_speed_data, sector_size, grid_size, color_white):
    font_scale = 0.5
    thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for y in range(grid_size[1]):
        for x in range(grid_size[0]):
            sector = (y, x)
            avg_speed = sector_speed_data[sector]['avg_vector']
            label = f"{avg_speed:.2f}"
            sector_center_x = int((x + 0.5) * sector_size[0])
            sector_center_y = int((y + 0.5) * sector_size[1])
            
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            text_x = int(sector_center_x - text_size[0] / 2)
            text_y = int(sector_center_y + text_size[1] / 2)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, color_white, thickness)  

def reset_sector_speed_data(sector_speed_data):
    # Пробегаем по всему массиву с помощью индексов
    for y in range(sector_speed_data.shape[0]):
        for x in range(sector_speed_data.shape[1]):
            sector_speed_data[y, x]['count'] = 0

def update_sector_speed_data(sector, norm_vector_length, sector_speed_data, count_reset):
    sector_data = sector_speed_data[sector]
    count = sector_data['count'] + 1
    # Если в одном из секторов count достигает 300, сбрасываем все значения
    if count >= count_reset:
        reset_sector_speed_data(sector_speed_data)
        return  # После сброса не обновляем текущий сектор
    # Обновляем данные для сектора
    sector_data['avg_vector'] = norm_vector_length
    sector_data['count'] = count


def write_data(video_path, grid_size, sector_speed_data):
    def serialize_data(data):
        if isinstance(data, dict):
            return {str(k): serialize_data(v) for k, v in data.items()}
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data

    serialized_data = serialize_data(sector_speed_data)
    data = {
        video_path: {
            'grid_size': grid_size,
            'sector_speed_data': serialized_data
        }
    }
    with open("data.json", 'w') as file:
        json.dump(data, file, indent=4)

def draw_trajectory(frame, track, N, color_blue):
    if len(track) < 2:
        return

    track_list = np.array(track)
    smoothed_track = []
    
    for i in range(len(track_list)):
        window = track_list[max(0, i - N + 1):i + 1]
        if len(window) > 1:
            avg_pos = np.mean(window, axis=0)
            smoothed_track.append(tuple(avg_pos))
    
    smoothed_track = np.array(smoothed_track)
    if len(smoothed_track) < 2:
        return

    for i in range(1, len(smoothed_track)):
        start_point = tuple(smoothed_track[i-1].astype(int))
        end_point = tuple(smoothed_track[i].astype(int))
        cv2.line(frame, start_point, end_point, color_blue, 2)


def draw_direction_vector(frame, track, N, color_yellow, bbox):
    if len(track) < 2:
        return None

    track_list = np.array(track)
    bbox_diagonal = np.sqrt(bbox[2] ** 2 + bbox[3] ** 2)
    vector_length = bbox_diagonal / 2

    directions = []

    for i in range(len(track_list)):
        window = track_list[max(0, i - N + 1):i + 1]
        if len(window) > 1:
            dx, dy = window[-1] - window[-2]
            speed = np.sqrt(dx ** 2 + dy ** 2)
            if speed != 0:
                direction = (dx / speed, dy / speed)
                directions.append(direction)

    if directions:
        avg_direction = np.mean(directions, axis=0)
        avg_direction = (avg_direction * vector_length).astype(int)
        prev_point = tuple(track_list[-2].astype(int))
        end_point = tuple(prev_point + avg_direction)
        # cv2.arrowedLine(frame, prev_point, end_point, color_yellow, 2)
        
        # Нормированная амплитуда
        A = 2 * np.linalg.norm(avg_direction) / bbox_diagonal
        # Угол направления в радианах
        phi = np.arctan2(avg_direction[1], avg_direction[0])
        # Преобразование угла в градусы
        phi_degrees = np.degrees(phi)
        # Приведение угла к диапазону [0, 360)
        # phi_degrees = (phi_degrees + 360) % 360
        # Приведение угла к диапазону от -180 до 180
        if phi_degrees > 180:
            phi_degrees -= 360
        
        # Комплексное число в форме A * exp(i*phi)
        complex_vector = A * np.exp(1j * phi)  # complex_vector = A * (np.cos(phi) + 1j * np.sin(phi))
        # print("A =", A, "phi (radians) =", phi, "phi (degrees) =", phi_degrees)
        return A, phi_degrees
    return 0, 0

polygon_points = []
def select_polygon_area(cap, frame_size, drawing):
    global polygon_points  # Не забудьте объявить global, если это необходимо

    if not drawing:
        print("Обработка видео без выделения полигона.")
        return []

    # Функция для обработки событий мыши
    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            print(f"Осталось точек: {len(polygon_points)}")  # Выводим количество оставшихся точек
            cv2.destroyAllWindows()  # Закрываем окно, если рисование завершено

    # Считываем первый кадр и позволяем пользователю выделить область
    success, frame = cap.read()
    if not success:
        print("Не удалось прочитать видео.")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    frame = cv2.resize(frame, frame_size)

    # Устанавливаем обработчик событий мыши
    cv2.namedWindow("Select Area")
    cv2.setMouseCallback("Select Area", draw_polygon)

    # Основной цикл для выбора области
    while True:
        temp_frame = frame.copy()
        if len(polygon_points) > 0:
            cv2.polylines(temp_frame, [np.array(polygon_points)], isClosed=False, color=(255, 0, 0), thickness=2)
        cv2.imshow("Select Area", temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return polygon_points


def draw_bounding_box(frame, track_id, elapsed_time, norm_vector_length, bbox_coords, color):
    x, y, w, h = bbox_coords
    cv2.putText(frame,
                f"{track_id} {elapsed_time:.2f}s",
                # f"{track_id} {elapsed_time:.2f} {norm_vector_length:.2f}",
                # f"{track_id}",
                (int(x - w / 2), int(y - h / 2) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)