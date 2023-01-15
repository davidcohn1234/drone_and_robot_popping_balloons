from yolo_object_detection import YoloObjectDetection
import numpy as np
import cv2
from robot_modules import Robot
import queue
import threading

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
        self.radius_to_follow = 150
        self.radius_to_get_away = 250
        self.forward_speed = 0.3
        self.attack_speed = 0.7
        self.robot_forward_speed = None
        self.robot_right_speed = None
        self.robot_yaw_speed = None
        self.vx = None
        self.vy = None

    def init_robo_cam(self):
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        # self.q = multiprocessing.Queue()
        # t=multiprocessing.Process(target=self._reader,)
        t.daemon = True
        t.start()

    def _reader(self):
        frame_id=0
        while True:
            rgb_image = self.robo_camera.read_cv2_image(strategy="newest")  # get frame by frame from tello
            objects_data = self.yolo_balloon_detection.detect_objects_in_frame(rgb_image, self.model)

            try:
                objects = centroid_tracker.update(x_balloons_centers_list, y_balloons_centers_list, balloon_colors_numbers_list,balloons_radiuses_list,x_offsets_list,y_offsets_list,frame_id)
            except:
                print('centroid failed')
            data = {'objects': objects}
            data['frame'] = frame
            data['frame_id'] = frame_id
            data['frame_with_ballons_data'] = frame_with_ballons_data

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(data)
            frame_id+=1

    def read(self):
        return self.q.get()


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

        pass

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

    def track(self, rgb_image, frame_index):
        objects_data = self.yolo_balloon_detection.detect_objects_in_frame(rgb_image, self.model)
        self.calculate_robot_data(objects_data=objects_data)
        if self.work_with_real_robot:
            self.robot.drive_speed(self.robot_forward_speed, self.robot_right_speed, self.robot_yaw_speed)
        rgb_image_with_ballons_data = self.yolo_balloon_detection.create_frame_with_objects_data(rgb_image, objects_data, frame_index)
        rgb_image_with_ballons_data = self.plot_objects_data(objects_data, rgb_image_with_ballons_data)
        self.save_image(rgb_image_with_ballons_data, frame_index)
        return rgb_image_with_ballons_data

