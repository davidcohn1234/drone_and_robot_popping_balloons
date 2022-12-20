import cv2
import torch
import os
import numpy as np

# greenLower = (42, 101, 96)
# greenUpper = (83, 168, 164)
#
# redLower = (0, 168, 133)
# redUpper = (4, 204, 161)
#
# blueLower = (100, 165, 110)
# blueUpper = (122, 216, 142)
#
# orangeLower = (5, 196, 156)
# orangeUpper = (30, 237, 188)
#
# purpleLower = (122, 100, 106)
# purpleUpper = (147, 139, 160)

min_blue, min_green, min_red = 165, 132, 140
max_blue, max_green, max_red = 203, 225, 222

green_color_info = {"name": 'green', "lower": (42, 101, 96), "upper": (83, 168, 164), "rgb_color": (0, 255, 0)}
#red_color_info = {"name": 'red', "lower": (0, 168, 133) , "upper": (5, 235, 219), "rgb_color": (0, 0, 255)}
red_color_info = {"name": 'red', "lower": (165, 132, 140) , "upper": (203, 225, 222), "rgb_color": (0, 0, 255)}
#red_color_info = {"name": 'red', "lower": (163, 193, 150), "upper": (207, 233, 255), "rgb_color": (0, 0, 255)}
#red_color_info = {"name": 'red', "lower": (158, 62, 98), "upper": (215, 246, 249), "rgb_color": (0, 0, 255)}
blue_color_info = {"name": 'blue', "lower": (89, 129, 123), "upper": (165, 241, 255), "rgb_color": (255, 0, 0)}
orange_color_info = {"name": 'orange', "lower": (6, 58, 135), "upper": (26, 228, 255), "rgb_color": (0, 165, 255)}
purple_color_info = {"name": 'purple', "lower": (81, 78, 106), "upper": (152, 160, 173), "rgb_color": (153, 51, 102)}

colors_ranges_info = [green_color_info, red_color_info, blue_color_info, orange_color_info, purple_color_info]

unknown_color_info = {"name": 'unknown', "lower": (0, 0, 0), "upper": (0, 0, 0), "rgb_color": (0, 0, 0)}

input_file_name = 'balloons_video_ninja_room'
#input_file_name = 'drone_with_injector_blowing_up_balloons'
input_file_full_path = f'./input_data/{input_file_name}.mp4'
#vid = cv2.VideoCapture("rtsp://192.168.1.28:8901/live")  # For streaming links
vid = cv2.VideoCapture(input_file_full_path)
#vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
model.conf = 0.8

def count_pixels_in_color_range(colorLower, colorUpper, frame, code=cv2.COLOR_BGR2HSV):
    hsv_frame = cv2.cvtColor(frame, code)
    mask = cv2.inRange(hsv_frame, colorLower, colorUpper)
    num_of_pixels = cv2.countNonZero(mask)
    return num_of_pixels

def calc_dist_from_point_to_rectangular_box(point, np_lower, np_upper):
    x_min = np_lower[0]
    x_max = np_upper[0]
    y_min = np_lower[1]
    y_max = np_upper[1]
    z_min = np_lower[2]
    z_max = np_upper[2]
    x = point[0]
    y = point[1]
    z = point[2]
    x_closest_point_on_box = np.clip(x, x_min, x_max)
    y_closest_point_on_box = np.clip(y, y_min, y_max)
    z_closest_point_on_box = np.clip(z, z_min, z_max)
    closest_point_on_box = np.array((x_closest_point_on_box, y_closest_point_on_box, z_closest_point_on_box))
    dist = np.linalg.norm(point - closest_point_on_box)
    return dist

def detect_balloon_data(frame, sub_image):
    colorCode = cv2.COLOR_BGR2HSV
    num_of_colors = len(colors_ranges_info)
    colors_num_of_pixels = np.zeros([num_of_colors])
    for index, single_color_range_info in enumerate(colors_ranges_info):
        lower = single_color_range_info["lower"]
        upper = single_color_range_info["upper"]
        single_color_num_of_pixels = count_pixels_in_color_range(lower, upper, frame, colorCode)
        colors_num_of_pixels[index] = single_color_num_of_pixels
    max_index = colors_num_of_pixels.argmax()
    balloon_data = colors_ranges_info[max_index]
    if np.sum(colors_num_of_pixels) == 0:
        # distances = np.zeros([num_of_colors])
        # hsv_sub_image = cv2.cvtColor(sub_image, colorCode)
        # average_pixel = np.mean(hsv_sub_image, axis=(0, 1))
        # for index, single_color_range_info in enumerate(colors_ranges_info):
        #     np_lower = np.array(single_color_range_info["lower"])
        #     np_upper = np.array(single_color_range_info["upper"])
        #     dist_from_point_to_rectangular_box = calc_dist_from_point_to_rectangular_box(average_pixel, np_lower, np_upper)
        #     distances[index] = dist_from_point_to_rectangular_box
        # min_index = distances.argmin()
        # balloon_data = colors_ranges_info[min_index]
        return unknown_color_info
    return balloon_data



#model.cuda()
frame_index = 0
while True:
    frame_index += 1
    _, frame = vid.read()
    if frame is None:
        break
    [frame_height, frame_width, channels] = frame.shape
    results = model(frame)


    david = results.pandas()
    david1 = david.xyxy[0]
    david2 = david1.to_numpy()
    num_of_balloons = david2.shape[0]
    x_ratio = 0.2
    y_ratio = 0.2
    print(f'frame_index = {frame_index}')

    images_output_folder = './balloons_images_with_data'

    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)

    if frame_index == 261:
        david8 = 8
    frame_with_ballons_data = frame.copy()
    for balloon_index in range(0,num_of_balloons):
        mask_RGB = np.zeros([frame_height, frame_width, channels], dtype=frame.dtype)
        current_balloon_data = david2[balloon_index]
        x_min = current_balloon_data[0]
        y_min = current_balloon_data[1]
        x_max = current_balloon_data[2]
        y_max = current_balloon_data[3]
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        x_middle = int(0.5 * (x_min + x_max))
        y_middle = int(0.5 * (y_min + y_max))
        balloon_width = x_max - x_min
        balloon_height = y_max - y_min

        x_start = int(x_min + x_ratio * balloon_width)
        x_end = int(x_min + (1 - x_ratio) * balloon_width)

        y_start = int(y_min + y_ratio * balloon_height)
        y_end = int(y_min + (1 - y_ratio) * balloon_height)








        sub_image = frame[y_start : y_end, x_start : x_end, :]



        mask_RGB[y_start : y_end, x_start : x_end, :] = sub_image

        # if frame_index == 260:
        #     cv2.imshow('mask_RGB', mask_RGB)
        #     key = cv2.waitKey(0) & 0xFF


        balloon_data = detect_balloon_data(mask_RGB, sub_image)
        balloon_color_name = balloon_data["name"]
        balloon_color = balloon_data["rgb_color"]

        thickness = 2
        frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, balloon_color, thickness)

        font = cv2.FONT_HERSHEY_SIMPLEX
        balloonFontScale = 0.8
        balloonThickness = 2
        center = (x_middle, y_middle)
        cv2.putText(frame_with_ballons_data, f'{balloon_color_name}', center, font,
                    balloonFontScale, balloon_color, balloonThickness, cv2.LINE_AA)



        confidence = current_balloon_data[4]
        obj_class = current_balloon_data[5]
        obj_name = current_balloon_data[6]
        david5 = 5
    #cv2.imshow('Video Live IP cam', results.render()[0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 0, 255)
    thickness = 2
    org = (20, 50)
    cv2.putText(frame_with_ballons_data, f'{frame_index}', org, font, fontScale, color, thickness, cv2.LINE_AA)

    file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
    cv2.imwrite(file_full_path, frame_with_ballons_data)

    cv2.imshow('frame', frame_with_ballons_data)
    #print(results.pandas().xyxy[0])
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# from threading import Thread
# import cv2, time

# class ThreadedCamera(object):
#     def __init__(self, src=0):
#         self.capture = cv2.VideoCapture(src)
#         self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

#         # FPS = 1/X
#         # X = desired FPS
#         self.FPS = 1/30
#         self.FPS_MS = int(self.FPS * 1000)

#         # Start frame retrieval thread
#         self.thread = Thread(target=self.update, args=())
#         self.thread.daemon = True
#         self.thread.start()

#     def update(self):
#         while True:
#             if self.capture.isOpened():
#                 (self.status, self.frame) = self.capture.read()
#             time.sleep(self.FPS)

#     def show_frame(self):
#         results = model(self.frame)
#         results.xy
#         cv2.imshow('Video Live IP cam',results.render()[0])
#         # cv2.imshow('frame', self.frame)
#         key = cv2.waitKey(self.FPS_MS) & 0xFF
#         if key ==ord('q'):
#             return


# if __name__ == '__main__':
#     src = 'rtsp://192.168.1.28:8901/live'
#     threaded_camera = ThreadedCamera(src)
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device="cpu")
#     model.conf = 0.5
#     while True:
#         try:
#             threaded_camera.show_frame()
#         except AttributeError:
#             pass