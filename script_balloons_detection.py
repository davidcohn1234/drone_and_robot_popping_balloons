import cv2
import torch
import os
import numpy as np

def count_pixels_in_color_range(colorLower, colorUpper, frame, colorCode=cv2.COLOR_BGR2HSV):
    hsv_frame = cv2.cvtColor(frame, colorCode)
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

def detect_balloon_data(frame, current_balloon_data, colors_ranges_info):
    x_ratio = 0.2
    y_ratio = 0.2
    x_min = current_balloon_data[0]
    y_min = current_balloon_data[1]
    x_max = current_balloon_data[2]
    y_max = current_balloon_data[3]
    balloon_width = x_max - x_min
    balloon_height = y_max - y_min
    x_start = int(x_min + x_ratio * balloon_width)
    x_end = int(x_min + (1 - x_ratio) * balloon_width)

    y_start = int(y_min + y_ratio * balloon_height)
    y_end = int(y_min + (1 - y_ratio) * balloon_height)

    sub_image = frame[y_start: y_end, x_start: x_end, :]
    num_of_colors = len(colors_ranges_info)
    colors_num_of_pixels = np.zeros([num_of_colors])
    for index, single_color_range_info in enumerate(colors_ranges_info):
        lower = single_color_range_info["lower"]
        upper = single_color_range_info["upper"]
        single_color_num_of_pixels = count_pixels_in_color_range(lower, upper, sub_image, colorCode=cv2.COLOR_BGR2HSV)
        colors_num_of_pixels[index] = single_color_num_of_pixels
    max_index = colors_num_of_pixels.argmax()
    balloon_data = colors_ranges_info[max_index]
    if np.sum(colors_num_of_pixels) == 0:
        unknown_color_info = {"name": 'unknown', "lower": (0, 0, 0), "upper": (0, 0, 0), "rgb_color": (0, 0, 0)}
        return unknown_color_info
    return balloon_data

def detect_balloons_in_frame(frame, model, colors_ranges_info):
    results = model(frame)

    frame_pandas_results = results.pandas()
    balloons_dataframes = frame_pandas_results.xyxy[0]
    balloons_numpy_locations = balloons_dataframes.to_numpy()
    num_of_balloons = balloons_numpy_locations.shape[0]


    frame_with_ballons_data = frame.copy()
    rects = []
    for balloon_index in range(0, num_of_balloons):

        current_balloon_data = balloons_numpy_locations[balloon_index]

        x_min = current_balloon_data[0]
        y_min = current_balloon_data[1]
        x_max = current_balloon_data[2]
        y_max = current_balloon_data[3]

        single_balloon_rectangle = np.array([x_min, y_min, x_max, y_max])
        rects.append(single_balloon_rectangle.astype("int"))

        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        x_middle = int(0.5 * (x_min + x_max))
        y_middle = int(0.5 * (y_min + y_max))





        balloon_data = detect_balloon_data(frame, current_balloon_data, colors_ranges_info)
        balloon_color_name = balloon_data["name"]
        balloon_color = balloon_data["rgb_color"]


        cv2.rectangle(frame_with_ballons_data, start_point, end_point, balloon_color, thickness=2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        balloonFontScale = 0.8
        balloonThickness = 2
        center = (x_middle, y_middle)
        cv2.putText(frame_with_ballons_data, f'{balloon_color_name}', center, font,
                    balloonFontScale, balloon_color, balloonThickness, cv2.LINE_AA)

        confidence = current_balloon_data[4]
        obj_class = current_balloon_data[5]
        obj_name = current_balloon_data[6]
    return rects


def main():
    green_color_info = {"name": 'green', "lower": (42, 101, 96), "upper": (83, 168, 164), "rgb_color": (0, 255, 0)}
    red_color_info = {"name": 'red', "lower": (165, 132, 140), "upper": (203, 225, 222), "rgb_color": (0, 0, 255)}
    blue_color_info = {"name": 'blue', "lower": (89, 129, 123), "upper": (165, 241, 255), "rgb_color": (255, 0, 0)}
    orange_color_info = {"name": 'orange', "lower": (6, 58, 135), "upper": (26, 228, 255), "rgb_color": (0, 165, 255)}
    purple_color_info = {"name": 'purple', "lower": (81, 78, 106), "upper": (152, 160, 173), "rgb_color": (153, 51, 102)}
    colors_ranges_info = [green_color_info, red_color_info, blue_color_info, orange_color_info, purple_color_info]

    input_file_name = 'balloons_video_ninja_room.mp4'
    input_file_full_path = f'./input_data/videos/{input_file_name}'
    vid = cv2.VideoCapture(input_file_full_path)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
    model.conf = 0.8

    images_output_folder = './balloons_images_with_data'

    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)

    frame_index = 0
    while True:
        frame_index += 1
        _, frame = vid.read()
        if frame is None:
            break
        print(f'frame_index = {frame_index}')
        frame_with_ballons_data = detect_balloons_in_frame(frame, model, colors_ranges_info)

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

main()