import cv2
import numpy as np


class YoloObjectDetection:
    def __init__(self):
        self.green_color_info = {"name": 'green', "lower": (42, 101, 96), "upper": (83, 168, 164), "rgb_color": (0, 255, 0)}
        self.red_color_info = {"name": 'red', "lower": (165, 132, 140), "upper": (203, 225, 222), "rgb_color": (0, 0, 255)}
        self.blue_color_info = {"name": 'blue', "lower": (89, 129, 123), "upper": (165, 241, 255), "rgb_color": (255, 0, 0)}
        self.orange_color_info = {"name": 'orange', "lower": (6, 58, 135), "upper": (26, 228, 255),
                             "rgb_color": (0, 165, 255)}
        self.purple_color_info = {"name": 'purple', "lower": (81, 78, 106), "upper": (152, 160, 173),
                             "rgb_color": (153, 51, 102)}
        self.unknown_color_info = {"name": 'unknown', "lower": (0, 0, 0), "upper": (0, 0, 0), "rgb_color": (0, 0, 0)}
        self.colors_ranges_info = [self.green_color_info, self.red_color_info, self.blue_color_info, self.orange_color_info, self.purple_color_info]
        pass

    def count_pixels_in_color_range(self, colorLower, colorUpper, frame, colorCode=cv2.COLOR_BGR2HSV):
        hsv_frame = cv2.cvtColor(frame, colorCode)
        mask = cv2.inRange(hsv_frame, colorLower, colorUpper)
        num_of_pixels = cv2.countNonZero(mask)
        return num_of_pixels

    def detect_object_color_data(self, frame, single_object_yolo_data):
        x_ratio = 0.2
        y_ratio = 0.2
        x_min = single_object_yolo_data[0]
        y_min = single_object_yolo_data[1]
        x_max = single_object_yolo_data[2]
        y_max = single_object_yolo_data[3]
        object_width = x_max - x_min
        object_height = y_max - y_min
        x_start = int(x_min + x_ratio * object_width)
        x_end = int(x_min + (1 - x_ratio) * object_width)

        y_start = int(y_min + y_ratio * object_height)
        y_end = int(y_min + (1 - y_ratio) * object_height)

        sub_image = frame[y_start: y_end, x_start: x_end, :]
        num_of_colors = len(self.colors_ranges_info)
        colors_num_of_pixels = np.zeros([num_of_colors])
        for index, single_color_range_info in enumerate(self.colors_ranges_info):
            lower = single_color_range_info["lower"]
            upper = single_color_range_info["upper"]
            single_color_num_of_pixels = self.count_pixels_in_color_range(lower, upper, sub_image,
                                                                     colorCode=cv2.COLOR_BGR2HSV)
            colors_num_of_pixels[index] = single_color_num_of_pixels
        max_index = colors_num_of_pixels.argmax()
        object_color_data = self.colors_ranges_info[max_index]
        if np.sum(colors_num_of_pixels) == 0:
            return self.unknown_color_info
        return object_color_data

    def detect_objects_in_frame(self, frame, yolo_model):
        results = yolo_model(frame)

        frame_pandas_results = results.pandas()
        objects_dataframes = frame_pandas_results.xyxy[0]
        objects_numpy_yolo_data = objects_dataframes.to_numpy()
        num_of_objects = objects_numpy_yolo_data.shape[0]

        objects_data = []
        for object_index in range(0, num_of_objects):
            single_object_yolo_data = objects_numpy_yolo_data[object_index]

            x_min = single_object_yolo_data[0]
            y_min = single_object_yolo_data[1]
            x_max = single_object_yolo_data[2]
            y_max = single_object_yolo_data[3]
            confidence = single_object_yolo_data[4]
            obj_class = single_object_yolo_data[5]
            obj_name = single_object_yolo_data[6]

            single_object_bounding_box = np.array([x_min, y_min, x_max, y_max]).astype("int")

            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))
            x_middle = int(0.5 * (x_min + x_max))
            y_middle = int(0.5 * (y_min + y_max))
            center_point = (x_middle, y_middle)

            single_object_color_data = self.detect_object_color_data(frame, single_object_yolo_data)
            single_object_data = dict()
            single_object_data['bounding_box'] = single_object_bounding_box
            single_object_data['start_point'] = start_point
            single_object_data['end_point'] = end_point
            single_object_data['start_point'] = start_point
            single_object_data['center_point'] = center_point
            single_object_data['confidence'] = confidence
            single_object_data['obj_class'] = obj_class
            single_object_data['obj_name'] = obj_name
            single_object_data['color_data'] = single_object_color_data
            objects_data.append(single_object_data)
        return objects_data

    def create_frame_with_objects_data(self, rgb_image, objects_data):
        rgb_image_with_objects_data = rgb_image.copy()
        num_of_objects_in_frame = len(objects_data)
        for object_index in range(0, num_of_objects_in_frame):
            single_object_data = objects_data[object_index]
            single_object_color_data = single_object_data['color_data']
            object_color_name = single_object_color_data["name"]
            object_color = single_object_color_data["rgb_color"]
            start_point = single_object_data['start_point']
            end_point = single_object_data['end_point']
            center_point = single_object_data['center_point']


            cv2.rectangle(rgb_image_with_objects_data, start_point, end_point, object_color, thickness=2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            objectFontScale = 0.8
            objectThickness = 2
            cv2.putText(rgb_image_with_objects_data, f'{object_color_name}', center_point, font,
                        objectFontScale, object_color, objectThickness, cv2.LINE_AA)
        return rgb_image_with_objects_data