import json
import cv2
import numpy as np

class JsonFileProcessor:
    def __init__(self, video_path, regions_file, frame_size=(1280, 720), makeJson=1, region_count=10):
        self.video_path = video_path
        self.regions_file = regions_file
        self.frame_size = frame_size
        self.makeJson = makeJson
        self.region_count = region_count
        self.COLORS = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
            (255, 0, 255), (128, 128, 0), (128, 0, 128), (0, 128, 128), (64, 64, 64)
        ]

        self.region_names = [f"Region_{i+1}" for i in range(self.region_count)]

    def initialize_json_file(self):
        """Инициализация JSON файла с исходными данными видео."""
        initial_data = {
            "url": self.video_path,
            "height": self.frame_size[1],
            "width": self.frame_size[0],
            "shapes": []
        }
        with open(self.regions_file, 'w') as file:
            json.dump(initial_data, file, indent=4)
        print(f"JSON файл {self.regions_file} инициализирован с размерами {self.frame_size[0]}x{self.frame_size[1]}.")

    # def add_polygon_to_json(self, region_name, points):
    def add_polygon_to_json(self, region_name, points, dwell_time):
        """Добавление полигона в JSON файл."""
        try:
            with open(self.regions_file, 'r') as file:
                data = json.load(file)

            shape = {
                "label": region_name,
                "points": [[float(x), float(y)] for x, y in points],
                "group_id": None,
                # "flags": {"dwell_time": dwell_time}  # Используем словарь
                "flags": dwell_time  # Преобразуем число в строку
            }
            data["shapes"].append(shape)

            with open(self.regions_file, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Полигон {region_name} успешно сохранён.")
        except Exception as e:
            print(f"Ошибка при сохранении полигона {region_name}: {e}")

    
    def select_polygon_area(self, cap):
        """Выбор области на видео."""
        polygon_points = [[] for _ in range(self.region_count)]
        current_region = 0

        def draw_polygon(event, x, y, flags, param):
            nonlocal current_region
            if event == cv2.EVENT_LBUTTONDOWN:
                # Добавление точки в текущий многоугольник
                polygon_points[current_region].append((x, y))

            elif event == cv2.EVENT_RBUTTONDOWN:
                # Замыкание полигона при правом клике
                if len(polygon_points[current_region]) > 2:
                    polygon_points[current_region].append(polygon_points[current_region][0])  # Замкнуть полигон

                    # Отобразить полигон с замкнутыми линиями
                    # temp_frame = frame.copy()
                    color = self.COLORS[current_region % len(self.COLORS)]

                    for i in range(1, len(polygon_points[current_region])):
                        cv2.line(temp_frame, polygon_points[current_region][i-1], polygon_points[current_region][i], color, 2)

                    # Замкнуть линию, если еще не было сделано
                    cv2.line(temp_frame, polygon_points[current_region][-2], polygon_points[current_region][0], color, 2)

                    centroid = np.mean(polygon_points[current_region], axis=0).astype(int)
                    # cv2.putText(temp_frame, self.region_names[current_region], tuple(centroid), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Отобразить замкнутый полигон на экране
                    cv2.imshow("Select Area", temp_frame)
                    cv2.waitKey(1)

                    # Теперь запросим время стоянки
                    try:
                        time_input = int(input(f"Введите время стоянки для {self.region_names[current_region]} (в секундах): "))
                    except ValueError:
                        print("Некорректный ввод времени. Установлено значение по умолчанию: 0 секунд.")
                        time_input = 0

                    # Сохраняем полигон в JSON только после запроса времени
                    self.add_polygon_to_json(
                        self.region_names[current_region],
                        polygon_points[current_region],
                        time_input
                    )
                    current_region += 1
                    if current_region >= self.region_count:
                        cv2.destroyAllWindows()

        success, frame = cap.read()
        if not success:
            print("Не удалось загрузить видео.")
            return []

        frame = cv2.resize(frame, self.frame_size)
        cv2.namedWindow("Select Area")
        cv2.setMouseCallback("Select Area", draw_polygon)

        while current_region < self.region_count:
            temp_frame = frame.copy()

            for idx, points in enumerate(polygon_points):
                if points:
                    color = self.COLORS[idx % len(self.COLORS)]

                    # Рисуем только линии между точками, пока полигон не замкнут
                    for i in range(1, len(points)):
                        cv2.line(temp_frame, points[i-1], points[i], color, 2)

                    centroid = np.mean(points, axis=0).astype(int)
                    cv2.putText(temp_frame, self.region_names[idx], tuple(centroid), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Select Area", temp_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Завершение программы по нажатию 'q'.")
                break

        cv2.destroyAllWindows()
        return polygon_points if current_region == self.region_count else None

    def load_json_to_dict(self):
        """Загружает JSON файл и возвращает его содержимое в виде словаря."""
        try:
            with open(self.regions_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Ошибка при чтении файла {self.regions_file}: {e}")
            return {}

    def rescale_points(self, points, original_size, target_size):
        """Масштабирует координаты точек из оригинального размера в целевой размер."""
        scale_x, scale_y = target_size[0] / original_size[0], target_size[1] / original_size[1]
        return [(int(x * scale_x), int(y * scale_y)) for x, y in points]

    def draw_polygon_and_label(self, frame, points, label):
        """Рисует полигон и отображает метку с динамически выбранным цветом."""
        # Определяем индекс региона из имени (например, "Region_1" -> 0)
        try:
            region_index = int(label.split('_')[-1]) - 1  # Преобразуем номер региона
        except ValueError:
            region_index = 0  # Если что-то пошло не так, используем 0

        # Берём цвет из self.COLORS по индексу
        color = self.COLORS[region_index % len(self.COLORS)]
        
        # Рисуем полигон
        points = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        # Подписываем регион
        centroid = np.mean(points, axis=0).astype(int)[0]
        cv2.putText(frame, label, tuple(centroid), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



    def process(self):
        """Основной метод для обработки видео и JSON файла."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Не удалось открыть видео.")
            exit()

        if self.makeJson == 1:
            self.initialize_json_file()
            self.select_polygon_area(cap)
            print("Процесс создания JSON завершён.")
        else:
            data = self.load_json_to_dict()
            if not data:
                exit("Файл JSON пуст или недоступен.")
            
            original_size = (data.get("width", 1280), data.get("height", 720))
            shapes = data.get("shapes", [])

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, original_size)
                for shape in shapes:
                    label = shape.get("label", "Unknown")
                    points = shape.get("points", [])
                    if points:
                        self.draw_polygon_and_label(frame, points, label)

                cv2.imshow("Regions Viewer", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print("Процесс завершён.")

        cap.release()
        cv2.destroyAllWindows()

# Пример использования
# video_path = '3.6.mp4'
# video_path = '17.mp4'
# video_path = '021.mp4'
# video_path = '2.avi'
# regions_file = 'regions.json'
# frame_size = (1280, 720)
# regions=5
# make_json = 1  # 1 - создание JSON, 0 - чтение и отображение JSON

# processor = JsonFileProcessor(video_path, regions_file, frame_size, makeJson=make_json, region_count=regions)
# processor.process()
