# from sys import _current_frames
# from turtle import forward, speed
import math
from Tracker import Tracker
# from multiprocessing import Process
#import yolo_ballons_detect as ybd
import yolo_ballons_detect1 as ybd1
from centroidtracker import CentroidTracker #from https://github.com/lev1khachatryan/Centroid-Based_Object_Tracking/tree/master/Python
centroid_tracker = CentroidTracker()
import sys
import time
import cv2
import math
import random
import queue
import threading
import numpy as np
import multiprocessing
from datetime import datetime
import csv
import os
import RoboNinja
import glob


now = datetime.now()
now = now.strftime("%H_%M_%S_%f")
images_output_folder = './balloons_images_with_data/robo/' + now
isExist = os.path.exists(images_output_folder)
if not isExist:
    os.makedirs(images_output_folder)
    os.makedirs(images_output_folder + '_orig_frame_')

our_yolo=1
logger_on=0
show_flag=1

FRAME_HEIGHT = 720
FRAME_WIDTH = 1280

FORWARD_SPEED =.3
ATTACK_SPEED =.7


RADIUS_TO_FOLLOW = 150
RADIUS_TO_GET_AWAY = 250
max_aim_radius=60 #distance of xoff and yoff from fram center
max_off_dist = 100




# """Colors of Baloon"""
# blue_color_info = {"name": 'blue', "lower": (42, 123, 66), "upper": (156, 255, 255), "rgb_color": (255, 0, 0)} # B G R # 69 172 66 144 255 255
# green_color_info = {"name": 'green', "lower": (39, 170, 64), "upper": (97, 255, 255), "rgb_color": (0, 255, 0)} #52 145 75 69 255 255
# red_color_info = {"name": 'red', "lower": (0, 150, 69) , "upper": (17, 255, 174), "rgb_color": (0, 0, 255)} #0 150 075 11 255 167  / 0 171 52 014 255 123
# yellow_color_info = {"name": 'yellow', "lower": (18, 166, 112), "upper": (48, 255, 255), "rgb_color": (0, 255, 255)} #28 172 122 48 255 255
# red_color_info2 = {"name": 'red', "lower": (161, 70, 69) , "upper": (255, 255, 255), "rgb_color": (0, 0, 255)} #138 91 74 255 255 255
#

#lower and upper for green are actually for yellow. it is temporary
greenLower = (28, 172, 122)
greenUpper = (48, 255, 255)
greenCode = cv2.COLOR_RGB2HSV

redLower = (0, 171, 52)
redUpper = (14, 255, 123)
redCode = cv2.COLOR_BGR2HSV

red2Lower = (138, 91, 74)
red2Upper = (255, 255, 255)
red2Code = cv2.COLOR_BGR2HSV

blueLower = (69 ,172, 66)
blueUpper = (144, 255, 255)
blueCode = cv2.COLOR_BGR2HSV

#lower and upper for yellow are actually for green. it is temporary
yellowLower = (52, 145, 75)
yellowUpper = (69, 255, 255)
yellowCode = cv2.COLOR_BGR2HSV

class Robo_baloon(object):
    def __init__(self):
        self.use_real_robot = False
        self.move_robot = False
        self.yaw_speed = 0
        self.forward_backward = 0
        self.left_right = 0


        if self.use_real_robot:
            self.takeOff = False
            self.ninja1 = RoboNinja.NinjaRobot('')
            self.ninja1.stopMove()
            self.robo_chassis = self.ninja1.chassis
            self.robo_arm = self.ninja1.arm
            self.robo_camera = self.ninja1.camera
            self.init_robo_cam()
            self.ninja1.stopMove()
        else:
            self.simulated_frame_index = 0
            self.images_folder = './input_data/images/david_house_01'

        # self.init_drone()
        self.init_PID()
        # The key is the index, the first element in the list is the progress level, the rest are color settings
        self.popBaloons = {0: [0, greenLower, greenUpper, greenCode], 1: [0, blueLower, blueUpper, blueCode],
                           2: [0, redLower, redUpper, redCode],3: [0, red2Lower, red2Upper, red2Code],4: [0, yellowLower, yellowUpper, yellowCode]}
        self.popCounter = 0
        self.colortracker = Tracker(FRAME_HEIGHT, FRAME_WIDTH,
                                    self.popBaloons[0][1], self.popBaloons[0][2], self.popBaloons[0][3])
        self.run__hunt()

    def init_robo_cam(self):
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        # self.q = multiprocessing.Queue()
        # t=multiprocessing.Process(target=self._reader,)
        t.daemon = True
        t.start()

    def read_simulated_frame(self):
        print(f'self.simulated_frame_index = {self.simulated_frame_index}')
        if self.simulated_frame_index == 94:
            david = 5
        frame_name = "{:05d}.jpg".format(self.simulated_frame_index)
        frame_full_path = os.path.join(self.images_folder, frame_name)
        frame = cv2.imread(frame_full_path)
        x_cent, y_cent, balloon_color, radius, xoffset, yoffset, frame_with_ballons_data = ybd1.run_script(frame, 0, self.simulated_frame_index)

        objects = centroid_tracker.update(x_cent, y_cent, balloon_color, radius, xoffset, yoffset, self.simulated_frame_index)

        print('centroid failed')
        data = {'objects': objects}
        data['frame'] = frame
        data['frame_id'] = self.simulated_frame_index
        data['frame_with_ballons_data'] = frame_with_ballons_data
        return data

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        frame_id=0
        while True:
            frame = self.robo_camera.read_cv2_image(strategy="newest")  # get frame by frame from tello
            x_balloons_centers_list,\
                y_balloons_centers_list,\
                balloon_colors_numbers_list,\
                balloons_radiuses_list,\
                x_offsets_list,\
                y_offsets_list, \
                frame_with_ballons_data = \
                ybd1.run_script(frame,0,frame_id)

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

    def init_PID(self):
        """
        We want to use a feedback control loop to direct the drone towards the recognized object from tracker.py
        to do this we will use a PID controller.
        find an optimal Vx and Vy using a PID
        distance = 100 # setpoint, pixels
        Vx = 0      #manipulated variable
        Vy = 0      #manipulated variable
        """

        def proportional():
            Vx = 0
            Vy = 0
            prev_time = time.time()
            Ix = 0
            Iy = 0
            ex_prev = 0
            ey_prev = 0
            Px = 0
            Py = 0
            Dx = 0
            Dy = 0
            ex = 0
            ey = 0

            while True:
                # yield an x and y velocity from xoff, yoff, and distance
                xoff, yoff, distance, current_time = yield Vx, Vy, Px, Py, Ix, Iy, Dx, Dy, ex, ey

                # PID Calculations
                ex = xoff - distance
                ey = yoff - distance
                delta_t = current_time - prev_time

                # Control Equations, constants are adjusted as needed
                Px = 0.1 * ex + 0.1 * (self.off_dist < 100) * np.sign(ex) * (self.frame_inx > 5)
                Py = 0.5 * ey + 0.1 * (self.off_dist < 100) * np.sign(ey) * (self.frame_inx > 5)
                # Ix = Ix + -0.001*ex*delta_t#0.001
                # Iy = Iy + -0.001*ey*delta_t
                Ix = 0 * (ex + ex_prev) * (delta_t) / 2 #0.05
                Iy = 0 * (ey + ey_prev) * (delta_t) / 2 #0.05
                Dx = 0.05 * (ex - ex_prev) / (delta_t)
                Dy = 0.1 * (ey - ey_prev) / (delta_t)

                Vx = Px + Ix + Dx
                Vy = Py + Iy + Dy
                # if Vx < 10 and Vx > 5:
                #     Vx = 10
                # if Vx > -10 and Vx < -5:
                #     Vx = -10
                # if Vy < 10 and Vy > 5:
                #     Vy = 10
                # if Vy > -10 and Vy < -5:
                #     Vy = -10

                # Vy=0
                # update the stored data for the next iteration
                ex_prev = ex
                ey_prev = ey
                prev_time = current_time

        self.PID = proportional()
        self.PID.send(None)

    """ 
    In this function we get the x, y , radius of the balloon
        By this, we calculate the Vx,Vy using the PID. These Vx, Vy is the velocity
        in which the drone should move:
        if the Vx > 0 we move in yaw velocity to the right,
        if the Vx < 0 we move in yaw velocity to the left. 
        if the Vy > 0 we move up, if the Vy < 0, we move down.
        Note that if the balloon was detected we get a radius bigger than 0,
        It means that the balloon was found and we should move forward with FORWARD_SPEED,
        Until the radius of the balloon is RADIUS_TO_GET_AWAY. After this we go backward 60cm.
        There are 2 steps to pop the balloon: 
        1) To move forward until RADIUS_TO_GET_AWAY
        2) To move backward
        Then we repeat this again, we check if the radius is equal to zero,
        if yes, and step 2 was done, it means that there is no balloon with this color (the balloon was popped), and we set our
        Tracker to the next color definded in the dict popBaloons.
        Also we always return the cv2 image to display it in a new window
    """

    def process_frame(self, frame, current_time, now,objects):
        xoff=-10000
        yoff=-10000
        radius=[]
        stat_data = {'time': current_time, 'frame': self.frame_inx, 'x_off': xoff, 'y_off': yoff, 'radi': radius,
                     'vx': [], 'vy': [], 'Px': [], 'Py': [], 'Ix': [], 'Iy': [], 'Dx': [], 'Dy': [], 'ex': [],
                     'ey': []}

        image = frame  # convert frame to cv2 image and show
        distance = 0
        print(self.frame_inx)
        keylist = list(objects.keys())
        xpos=[]
        ypos = []
        radi=[]
        xoffset=[]
        yoffset=[]
        balloon_color=[]
        off_dist=[]
        off_dist1=[]
        if len(self.no_go_list) > 0:
            try:
                for no_go in self.no_go_list:
                    keylist.remove(no_go)
            except:
                pass

        if len(keylist)>0:
            no_ballon_flag = 0
            # for i in range(0, max(keylist) + 1):
            for i in keylist:
                try:
                    xpos.append(objects[i][0])
                    ypos.append(objects[i][1])
                    radi.append(objects[i][2])
                    xoffset.append(objects[i][3])
                    yoffset.append(objects[i][4])
                    off_dist.append(math.sqrt(objects[i][3] ** 2 + objects[i][4] ** 2))
                    balloon_color.append(objects[i][5])
                except:
                    pass
            if self.prev_ballon_indx>-1:
                try:
                    keylist_index = keylist.index(self.prev_ballon_indx)
                    print(self.no_go_counter)
                    if self.use_real_robot:
                        if  self.ninja1.get_robo_state()[1]>140:
                            self.no_go_counter+=1
                            if self.no_go_counter>5:
                                self.ninja1.stopMove()
                                self.no_go_list.append(self.prev_ballon_indx)
                                self.prev_ballon_indx=-1
                                self.no_go_counter=0
                                return image, stat_data, image
                except:
                    keylist_index=-1
                if keylist_index > -1:
                    xoff=xoffset[keylist_index]
                    yoff=yoffset[keylist_index]
                    radius=radi[keylist_index]
                    off_dist1=off_dist[keylist_index]

                else:
                    min_value = min(off_dist)
                    min_index = off_dist.index(min_value)
                    xoff = xoffset[min_index]
                    yoff = yoffset[min_index]
                    radius = radi[min_index]
                    self.prev_ballon_indx = keylist[min_index]
                    off_dist1=off_dist[min_index]

            else:
                min_value = min(off_dist)
                min_index = off_dist.index(min_value)
                off_dist1 = off_dist[min_index]
                xoff=xoffset[min_index]
                yoff=yoffset[min_index]
                radius=radi[min_index]
                self.prev_ballon_indx=keylist[min_index]
        else:
            no_ballon_flag=1
        # 'red'       is 1 in balloons_color
            # 'blue'      is 2 in balloons_color
            # 'green'     is 3 in balloons_color
            # 'yellow'    is 4 in balloons_color
            # 'other'     is 0 in balloons_color




        # # if 'Prev_Subimage_pos'  not in locals():
        # #     Prev_Subimage_pos=[]
        # prev_center_ballon = self.prev_center_ballon
        # off_dist = -0.25
        # no_ballon_flag = 0
        # move_factor=0
        # try:
        #     if self.Prev_Subimage_pos.__len__() > 0:
        #         Prev_Subimage_pos = self.Prev_Subimage_pos
        #         [frame_height, frame_width, channels] = image.shape
        #         mask_RGB = np.zeros([frame_height, frame_width, channels], dtype=image.dtype)
        #         balloon_width = Prev_Subimage_pos[1] - Prev_Subimage_pos[0]
        #         balloon_height = Prev_Subimage_pos[3] - Prev_Subimage_pos[2]
        #         x_ratio = 0.1
        #         y_ratio = 0.1
        #         x_start = int(Prev_Subimage_pos[0] - x_ratio * balloon_width)
        #         x_end = int(Prev_Subimage_pos[0] + (1 + x_ratio) * balloon_width)
        #         y_start = int(Prev_Subimage_pos[2] - y_ratio * balloon_height)
        #         y_end = int(Prev_Subimage_pos[2] + (1 + y_ratio) * balloon_height)
        #         if x_start <= 0:
        #             x_start = 0
        #         if y_start <= 0:
        #             y_start = 0
        #         if x_end > frame_width:
        #             x_end = frame_width
        #         if y_end > frame_height:
        #             y_end = frame_height
        #
        #         sub_image = frame[y_start: y_end, x_start: x_end, :]
        #         mask_RGB[y_start: y_end, x_start: x_end, :] = sub_image
        #         now = datetime.now()
        #         if 0:
        #             curr_time = now.strftime("%H_%M_%S_%f")
        #             images_output_folder = './balloons_images_with_data'
        #             file_full_path = "{}/sub_image_tellPop_{:s}.jpg".format(images_output_folder, '@' + curr_time)
        #             cv2.imwrite(file_full_path, mask_RGB)
        #         image = mask_RGB
        # except:
        #     print('cant calculate Prev_Subimage_pos')
        # xoff = int(-10000)
        # yoff = int(-10000)
        # #
        # if our_yolo:
        #
        #     xoff, yoff, radius, Prev_Subimage_pos, prev_center_ballon, frame_with_ballons_data, sub_image, full_frame = ybd.run_script(frame, image, self.frame_inx, prev_center_ballon, now,move_factor)
        #     self.Prev_Subimage_pos = Prev_Subimage_pos
        #     self.prev_center_ballon = prev_center_ballon
        #     image = self.colortracker.draw_arrows(frame_with_ballons_data, xoff,
        #                                           yoff)  # draw the arrows that shows where the drone should move to
        #     # ballon_ratio = (Prev_Subimage_pos[1] - Prev_Subimage_pos[0]) / (Prev_Subimage_pos[3] - Prev_Subimage_pos[2])
        #
        #     # if self.frame_inx<1:
        #     #     first_ballon_ratio=ballon_ratio
        #     # if ballon_ratio<0.95*first_ballon_ratio or ballon_ratio>1.05*first_ballon_ratio:
        #     #     if Prev_Subimage_pos[2]==0:
        #     #         new_min_y=Prev_Subimage_pos[3]-(Prev_Subimage_pos[1] - Prev_Subimage_pos[0]) /first_ballon_ratio
        #     #         y_cent=(Prev_Subimage_pos[3]-new_min_y)/2
        #     #         yoff
        #     #         pass
        #     #     elif Prev_Subimage_pos[3]==frame_height:
        #     #         pass
        #     #     elif Prev_Subimage_pos[0] == 0:
        #     #         pass
        #     #     elif Prev_Subimage_pos[1] == frame_width:
        #     #         pass
        # if not our_yolo or radius == 0:
        #     # xoff, yoff,radius = self.colortracker.track(image) # we use track function from colortracker object
        #     image = self.colortracker.draw_arrows(image, int(-10000),
        #                                           int(-10000))  # draw the arrows that shows where the drone should move to

        # off_dist = math.sqrt(xoff ** 2 + yoff ** 2)

        self.off_dist = off_dist1
        # todo create logic to keep same ballon in the frame until pops
        if xoff > -10000:
            self.center_off = math.sqrt(xoff ** 2 + yoff ** 2)
            Vx, Vy, Px, Py, Ix, Iy, Dx, Dy, ex, ey = self.PID.send(
                [xoff, yoff, distance, current_time])  # calculate the X,Y Velocity using PID
            stat_data = {'time': current_time, 'frame': self.frame_inx, 'x_off': xoff, 'y_off': yoff, 'radi': radius,
                         'vx': Vx, 'vy': Vy, 'Px': Px, 'Py': Py, 'Ix': Ix, 'Iy': Iy, 'Dx': Dx, 'Dy': Dy, 'ex': ex,
                         'ey': ey}
            # return image, stat_data, image,
        # Show the radius of the
        # drone in the cv2 image
        try:
            cv2.putText(image, f" off dist: {round(off_dist1, 2)}", (30, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)
        except:
            cv2.putText(image, f" No Balloons", (30, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 2)

        try:
            cv2.putText(image, f" vx: {round(Vx, 2)} , vy: {round(Vy, 2)}", (30, 150), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 0, 255), 2)
        except:
            cv2.putText(image, f" vx: {[]} , vy: {[]}", (30, 150), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        (0, 0, 255), 2)

        try:
            if Vx > 0:
                # if Vx<10:
                #     Vx=10
                # self.yaw_speed = int(abs(Vx)) # rotate right
                if Vx < 0:
                    self.left_right = int(abs(Vx))  # move right
                    self.yaw_speed = 0  # rotate right
                else:
                    self.left_right = 0  # move right
                    self.yaw_speed = int(abs(Vx))  # rotate right

            if Vx < 0:
                # if Vx>-10:
                #     Vx=-10
                # self.yaw_speed = - int(abs(Vx)) # rotate left
                if Vx > 0:
                    self.left_right = -int(abs(Vx))  # move right
                    self.yaw_speed = 0  # rotate right
                else:
                    self.left_right = 0  # move right
                    self.yaw_speed = -int(abs(Vx))  # rotate right

            if Vy > 0:
                # if Vy < 10 and Vy>5:
                #     Vy = 10

                self.up_down_speed = int(abs(Vy))  # go up

            if Vy < 0:
                # if Vy > -10 and Vy<-5:
                #     Vy = -10

                self.up_down_speed = - int(abs(Vy))  # go down

            """ These lines make the drone pop the balloon (he goes forward and then backward)
                Using the two steps mentioned above we know in which state the drone is
            """
            cv2.putText(image, f" radius: {radius}", (30, 70), cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 255), 2)

            if radius > 0 and radius < RADIUS_TO_FOLLOW and radius < max_aim_radius:  # if the drone is not close to the balloon
                self.forward_backward = FORWARD_SPEED  # set to him forward speed towards the balloon
                cv2.putText(image, f" IF: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)

            elif radius > 0 and radius < RADIUS_TO_FOLLOW and radius > max_aim_radius and off_dist1 > max_off_dist:  # if the drone is not close to the balloon
                if self.use_real_robot:
                    self.ninja1.stopMove()
                    self.ninja1.armUpdoun(self.up_down_speed)
                # self.forward_backward = -1  # set to him forward speed towards the balloon
                cv2.putText(image, f" radius > 0 and radius < RADIUS_TO_FOLLOW and radius > max_aim_radius and off_dist > max_off_dist: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)

            elif radius > 0 and radius < RADIUS_TO_FOLLOW and radius > max_aim_radius and off_dist1 < max_off_dist:  # if the drone is not close to the balloon
                self.forward_backward = FORWARD_SPEED  # set to him forward speed towards the balloon
                cv2.putText(image, f" radius > 0 and radius < RADIUS_TO_FOLLOW and radius > max_aim_radius and off_dist < max_off_dist: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)

            elif radius > 0 and radius >= RADIUS_TO_GET_AWAY:  # if the drone is too close to the balloon
                # time.sleep(0.1)
                self.forward_backward = -ATTACK_SPEED  # set to him forward speed towards the balloon
                if self.use_real_robot:
                    time.sleep(0.7)
                    self.ninja1.stopMove()
                cv2.putText(image, f" radius > 0 and radius >= RADIUS_TO_GET_AWAY: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)
                self.no_go_list.append(self.prev_ballon_indx)
                self.prev_ballon_indx = -1
            elif radius > 0 and radius >= RADIUS_TO_FOLLOW and off_dist1 < max_off_dist:  # if the drone is close to attack
                if self.move_robot:
                    self.ninja1.move_any(ATTACK_SPEED, 0, 0)
                    time.sleep(0.4)
                    self.ninja1.move_any(-ATTACK_SPEED, 0, 0)
                    time.sleep(0.7)
                    self.ninja1.stopMove()
                self.forward_backward = 0  # set to him forward speed towards the balloon
                cv2.putText(image, f" radius > 0 and radius >= RADIUS_TO_FOLLOW and off_dist < max_off_dist: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)
                self.no_go_list.append(self.prev_ballon_indx)
                self.prev_ballon_indx = -1
            elif radius == 0:
                if self.use_real_robot:
                    self.ninja1.stopMove()
            # elif radius > 0 and radius >= RADIUS_TO_FOLLOW and off_dist > max_off_dist:  # if the drone is close to attack
            #     self.ninja1.stopMove()
            #     # self.forward_backward = -1  # set to him forward speed towards the balloon
                cv2.putText(image, f" radius == 0: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)

            elif radius > 0:
                self.left_right = 0
                self.up_down_speed = 0
                self.yaw_speed = 0
                cv2.putText(image, f" radius > 0: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)
            else:
                if self.use_real_robot:
                    self.ninja1.stopMove()
                # self.forward_backward = -5
                # self.drone_move_back(-5)
                cv2.putText(image, f" else: ", (30, 500),cv2.FONT_HERSHEY_COMPLEX,0.6, (0, 0, 255), 2)

            image = self.colortracker.draw_arrows(image, xoff,yoff)  # draw the arrows that shows where the drone should move to

            cv2.putText(image, f" FORWARD_SPEED: {round(self.forward_backward, 2)}", (30, 200),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.putText(image, f" left_right_SPEED: {round(self.left_right, 2)}", (30, 230),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.putText(image, f" up_down_speed_SPEED: {round(self.up_down_speed, 2)}", (30, 260),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 255), 2)
            cv2.putText(image, f" Yaw_SPEED: {round(self.yaw_speed, 2)}", (30, 290), cv2.FONT_HERSHEY_COMPLEX,
                        0.6, (0, 0, 255), 2)
            # self.yaw_speed = 0
            if self.move_robot:
                self.ninja1.move_any(self.forward_backward, self.left_right, self.yaw_speed)
                if self.forward_backward<0:
                    time.sleep(.5)
                    self.ninja1.stopMove()
                # self.ninja1.armUpdoun(self.up_down_speed)
        except:
            if self.use_real_robot:
                if self.frame_inx > 1:
                    if not no_ballon_flag:
                        # self.ninja1.move_any(-1,0, 0)
                        # time.sleep(1)
                        self.ninja1.stopMove()
                        print('no_ballon_flag back')
                        no_ballon_flag = 1
                    else:
                        # self.ninja1.move_any(0, 0, 45)
                        # time.sleep(1)
                        self.ninja1.stopMove()
                        no_ballon_flag = 0
                        print('no_ballon_flag rotate_clockwise')

        return image, stat_data, image

    def run__hunt(self):
        if self.use_real_robot:
            self.ninja1.stopMove()
        frame_inx = 0
        if logger_on:
            csv_columns = ['time', 'frame', 'x_off', 'y_off', 'radi', 'vx', 'vy', 'Px', 'Py', 'Ix', 'Iy', 'Dx', 'Dy',
                           'ex',
                           'ey']

            now = datetime.now()
            current_time = now.strftime("%-d_%-m_%-y_%H_%M_%S")
            csv_file = "roboPop_log_" + current_time + ".csv"

            # with open(csv_file, 'w') as csvfile:
            csvfile = open(images_output_folder +'/' + csv_file, 'w')
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)

            writer.writeheader()
        self.frame_inx=0
        self.Prev_Subimage_pos = []
        self.prev_center_ballon = []
        self.prev_ballon_indx=-1
        counter=0
        self.no_go_list=[]
        self.no_go_counter=0
        if not self.use_real_robot:
            jpg_expression = self.images_folder + '/*.jpg'
            self.list_of_images = glob.glob(jpg_expression)
            self.num_of_images = len(self.list_of_images)

        while True:
            print(frame_inx)
            if self.use_real_robot:
                temp_info = self.read()  # get frame by frame from robo
            else:
                self.simulated_frame_index += 1
                if self.simulated_frame_index == self.num_of_images:
                    break
                else:
                    temp_info = self.read_simulated_frame()
            frame=temp_info["frame"]
            objects=temp_info["objects"]
            self.frame_inx=temp_info["frame_id"]
            frame_with_ballons_data = temp_info["frame_with_ballons_data"]


            frame_time = time.time()
            now = datetime.now()
            now = now.strftime("%H_%M_%S_%f")
            height, width, _ = frame.shape  # get the frame config
            current_frame = frame_with_ballons_data  # set the frame config
            current_frame, stat_data, full_frame = self.process_frame(current_frame, frame_time,now,objects)
            # self.ninja1.move_any(self.forward_backward, self.left_right, self.yaw_speed)
            # # time.sleep(.2)
            current_frame = cv2.resize(current_frame, (width, height))

            if logger_on:
                writer.writerow(stat_data)

            file_full_path = "{}/{:05d}.jpg".format(images_output_folder, self.frame_inx)
            cv2.imwrite(file_full_path, current_frame)

            file_full_path = "{}/{:05d}.jpg".format(images_output_folder + '_orig_frame_', self.frame_inx)
            cv2.imwrite(file_full_path, full_frame)

            cv2.putText(current_frame, f" {self.frame_inx}", (30, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6,(0, 0, 0), 2)

            if show_flag:
                cv2.imshow("Image", current_frame)  # show the last frame in a new window

            if cv2.waitKey(1) & 0xFF == ord('q'):  # and telloTrack.takeOff: # if the 'q' pressed on the cv2 screen
                break
            self.frame_inx += 1
            if not stat_data['x_off'] > -10000:
                if self.move_robot:
                    if counter>4:
                        counter=0
                        self.ninja1.move_any(0,0,35)
                        time.sleep(.7)
                        # self.ninja1.move_any(1, 0, 90)
                        # time.sleep(.3)
                        self.ninja1.move_any(0,0,1)
                        self.ninja1.stopMove()
                        time.sleep(1)
                        print('didnt find balloon - rotating ')
                    else:
                        counter+=1
def main():
    Robo_baloon()

if __name__ == "__main__":
    main()