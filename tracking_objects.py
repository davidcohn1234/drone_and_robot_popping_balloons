from yolo_object_detection import YoloObjectDetection
import numpy as np
import cv2
from robot_modules import Robot
from centroidtracker1 import CentroidTracker
import queue
import threading
import glob
import common_utils

class Tracker:
    def __init__(self, model, images_output_folder, image_height, image_width):
        self.model = model
        self.yolo_balloon_detection = YoloObjectDetection()
        self.images_output_folder = images_output_folder
        self.image_height = image_height
        self.image_width = image_width
        self.image_center_point = np.array((int(0.5 * image_width), int(0.5 * image_height)))
        self.work_with_real_robot = False
        if self.work_with_real_robot:
            self.robot = Robot()
            self.robo_camera = self.robot.camera
        else:
            self.simulated_frame_index = -1
            self.images_folder = './input_data/images/01_david_house'
            #self.images_folder = './input_data/images/03_balloons_video_ninja_room'
            jpg_expression = self.images_folder + '/*.jpg'
            self.list_of_images = sorted(glob.glob(jpg_expression))
            self.num_of_images = len(self.list_of_images)
        self.radius_to_follow = 150
        self.radius_to_get_away = 250
        self.forward_speed = 0.3
        self.attack_speed = 0.7
        self.robot_forward_speed = None
        self.robot_right_speed = None
        self.robot_yaw_speed = None
        self.vx = None
        self.vy = None
        self.main_output_folder = './balloons_images_with_color_name'
        self.images_output_folder = self.main_output_folder + '/' + 'images'
        common_utils.create_empty_output_folder(self.images_output_folder)
        self.ct = CentroidTracker()
        self.init_robo_cam()

    def init_robo_cam(self):
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        # self.q = multiprocessing.Queue()
        # t=multiprocessing.Process(target=self._reader,)
        t.daemon = True
        t.start()

    def _reader(self):
        frame_index = 0
        while True:
            if self.work_with_real_robot:
                rgb_image = self.robo_camera.read_cv2_image(strategy="newest")
            else:
                self.simulated_frame_index += 1
                if self.simulated_frame_index == self.num_of_images:
                    break
                else:
                    image_path = self.list_of_images[self.simulated_frame_index]
                    rgb_image = cv2.imread(image_path)
            objects_data = self.yolo_balloon_detection.detect_objects_in_frame(rgb_image, self.model)
            objects_ordered_dict = self.ct.update(objects_data)
            rgb_image_with_balloons_data = self.yolo_balloon_detection.create_frame_with_objects_data(rgb_image,
                                                                                                      objects_data,
                                                                                                      frame_index)

            data = dict()
            data['objects'] = objects_ordered_dict
            data['frame'] = rgb_image
            data['frame_index'] = frame_index
            data['rgb_image_with_balloons_data'] = rgb_image_with_balloons_data

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(data)
            frame_index += 1

    def read(self):
        data = self.q.get()
        return data


    def save_image(self, rgb_image, frame_index):
        file_full_path = "{}/{:05d}.jpg".format(self.images_output_folder, frame_index)
        cv2.imwrite(file_full_path, rgb_image)

    def calculate_robot_forward_speed(self, object_radius):
        if object_radius < self.radius_to_follow:
            self.robot_forward_speed = self.forward_speed
        elif self.radius_to_follow <= object_radius <= self.radius_to_get_away:
            self.robot_forward_speed = self.attack_speed
        else: #if object_radius > self.radius_to_get_away
            self.robot_forward_speed = -self.attack_speed
        return

    def calculate_robot_right_speed(self, object_radius):
        if self.vx < 0:
            self.robot_right_speed = int(abs(self.vx))  # move right
            self.robot_yaw_speed = 0  # rotate right
        else:
            self.robot_right_speed = 0  # move right
            self.robot_yaw_speed = int(abs(self.vx))  # rotate right
        return

    def calculate_robot_data(self, objects_data):
        for (object_index, single_object_data) in enumerate(objects_data):
            offset_from_image_center_to_object_center = single_object_data['offset_from_image_center_to_object_center']
            self.vx = offset_from_image_center_to_object_center[0]
            self.vy = offset_from_image_center_to_object_center[1]
            object_radius = single_object_data['radius']
            self.calculate_robot_forward_speed(object_radius)
            self.calculate_robot_right_speed(object_radius)

    def plot_objects_data(self, objects_data, rgb_image):
        rgb_image_with_data = rgb_image.copy()
        circle_radius = 8
        for (object_index, single_object_data) in enumerate(objects_data):
            object_center_point = single_object_data['center_point']
            #cv2.line(img=rgb_image_with_data, pt1=self.image_center_point, pt2=object_center_point, color=(255, 0, 0), thickness=5)
            cv2.arrowedLine(img=rgb_image_with_data, pt1=self.image_center_point, pt2=object_center_point, color=(255, 0, 0), thickness=5)
            cv2.circle(img=rgb_image_with_data, center=self.image_center_point, radius=circle_radius, color=(0, 255, 0), thickness=-1)
            cv2.circle(img=rgb_image_with_data, center=object_center_point, radius=circle_radius, color=(0, 255, 255),
                       thickness=-1)
        return rgb_image_with_data

    def track(self):
        # objects_data = self.yolo_balloon_detection.detect_objects_in_frame(rgb_image, self.model)
        # objects_ordered_dict = self.ct.update(objects_data)
        # self.calculate_robot_data(objects_data=objects_data)
        # if self.work_with_real_robot:
        #     self.robot.drive_speed(self.robot_forward_speed, self.robot_right_speed, self.robot_yaw_speed)
        # rgb_image_with_balloons_data = self.yolo_balloon_detection.create_frame_with_objects_data(rgb_image, objects_data, frame_index)
        # rgb_image_with_balloons_data = self.plot_objects_data(objects_data, rgb_image_with_balloons_data)
        # self.save_image(rgb_image_with_balloons_data, frame_index)
        # return rgb_image_with_balloons_data

        data = self.read()
        objects_ordered_dict = data['objects']
        rgb_image = data['frame']
        frame_index = data['frame_index']
        rgb_image_with_balloons_data = data['rgb_image_with_balloons_data']

        file_full_path = "{}/{:05d}.jpg".format(self.images_output_folder, frame_index)
        cv2.imwrite(file_full_path, rgb_image_with_balloons_data)
        cv2.imshow('rgb_image_with_balloons_data', rgb_image_with_balloons_data)
        key = cv2.waitKey(1) & 0xFF
        # if key == ord('q'):
        #     break

        return


