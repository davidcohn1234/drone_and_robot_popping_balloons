# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from centroidtracker1 import CentroidTracker
from centroidtrackerdebugger import CentroidTrackerDebugger
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
import os
import common_utils
import glob


def count_pixels_in_color_range(colorLower, colorUpper, frame, colorCode=cv2.COLOR_BGR2HSV):
    hsv_frame = cv2.cvtColor(frame, colorCode)
    mask = cv2.inRange(hsv_frame, colorLower, colorUpper)
    num_of_pixels = cv2.countNonZero(mask)
    return num_of_pixels


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

        x_min = round(current_balloon_data[0])
        y_min = round(current_balloon_data[1])
        x_max = round(current_balloon_data[2])
        y_max = round(current_balloon_data[3])

        single_balloon_rectangle = np.array([x_min, y_min, x_max, y_max], int)
        rects.append(single_balloon_rectangle)

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

def create_empty_output_folder(images_output_folder):
    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)
    else:
        files = glob.glob(images_output_folder + '/*.jpg')
        for f in files:
            os.remove(f)


def create_image_with_balloons_data(rgb_image, bounding_boxes, color):
    rgb_image_with_data = rgb_image.copy()
    num_of_bounding_boxes = len(bounding_boxes)
    for index in range(0, num_of_bounding_boxes):
        current_bounding_box = bounding_boxes[index]

        x_min = round(current_bounding_box[0])
        y_min = round(current_bounding_box[1])
        x_max = round(current_bounding_box[2])
        y_max = round(current_bounding_box[3])

        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))

        x_center = round(0.5 * (x_min + x_max))
        y_center = round(0.5 * (y_min + y_max))

        centroid = np.array((int(x_center), int(y_center)))
        # coordinateText = "({}, {})".format(x_center, y_center)
        # cv2.putText(rgb_image_with_data, coordinateText, (centroid[0], centroid[1] + 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.circle(rgb_image_with_data, (centroid[0], centroid[1]), 8, color, -1)
        cv2.rectangle(rgb_image_with_data, start_point, end_point, color, thickness=2)
    return rgb_image_with_data

def plot_explanation_for_centroid_tracker(all_frames_data_list):
    frame_1_data = all_frames_data_list[0]
    frame_2_data = all_frames_data_list[1]

    rgb_image_1 = frame_1_data['rgb_image']
    rgb_image_2 = frame_2_data['rgb_image']

    ratio_image_1 = 0.5
    combined_rgb = (ratio_image_1 * rgb_image_1 + (1 - ratio_image_1) * rgb_image_2).astype(np.uint8)
    bounding_boxes_frame_1 = frame_1_data['balloons_bounding_boxes']
    combined_rgb_with_frame_1_data = create_image_with_balloons_data(combined_rgb, bounding_boxes_frame_1, color=(255, 0, 0))

    bounding_boxes_frame_2 = frame_2_data['balloons_bounding_boxes']
    combined_rgb_with_frames_data = create_image_with_balloons_data(combined_rgb_with_frame_1_data, bounding_boxes_frame_2,
                                                                     color=(0, 0, 255))

    cv2.imshow('combined_rgb_with_frames_data', combined_rgb_with_frames_data)
    cv2.waitKey(0)

def save_frame(rgb_image_with_id_data, images_output_folder, frame_index):
    file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
    cv2.imwrite(file_full_path, rgb_image_with_id_data)

def main():
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    ct_debugger = CentroidTrackerDebugger()

    folder_name = '03_balloons_video_ninja_room'
    input_folder_full_path = f'./input_data/images/' + folder_name
    min_frame_index = 47
    max_frame_index = 50
    # input_file_name = range(min_frame_index, max_frame_index + 1)
    # image_full_path = input_folder_full_path + '/' + input_file_name
    jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
    #jpg_files = jpg_files[min_frame_index:max_frame_index+1]
    #jpg_files = np.array(jpg_files)[[min_frame_index, max_frame_index]]
    # jpg_files = [image_full_path]
    frame_milliseconds = 1


    time.sleep(2.0)

    green_color_info = {"name": 'green', "lower": (42, 101, 96), "upper": (83, 168, 164), "rgb_color": (0, 255, 0)}
    red_color_info = {"name": 'red', "lower": (165, 132, 140), "upper": (203, 225, 222), "rgb_color": (0, 0, 255)}
    blue_color_info = {"name": 'blue', "lower": (89, 129, 123), "upper": (165, 241, 255), "rgb_color": (255, 0, 0)}
    orange_color_info = {"name": 'orange', "lower": (6, 58, 135), "upper": (26, 228, 255), "rgb_color": (0, 165, 255)}
    purple_color_info = {"name": 'purple', "lower": (81, 78, 106), "upper": (152, 160, 173), "rgb_color": (153, 51, 102)}
    colors_ranges_info = [green_color_info, red_color_info, blue_color_info, orange_color_info, purple_color_info]

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
    model.conf = 0.8


    main_tracking_output_folder = './balloons_images_with_tracking_data'
    images_tracking_output_folder = main_tracking_output_folder + '/' + 'images'
    videos_tracking_output_folder = main_tracking_output_folder + '/' + 'videos'
    create_empty_output_folder(images_tracking_output_folder)
    create_empty_output_folder(videos_tracking_output_folder)

    main_id_output_folder = './balloons_images_with_id_data'
    images_id_output_folder = main_id_output_folder + '/' + 'images'
    videos_id_output_folder = main_id_output_folder + '/' + 'videos'
    create_empty_output_folder(images_id_output_folder)
    create_empty_output_folder(videos_id_output_folder)


    debug_mode = True

    # loop over the frames from the video stream
    if debug_mode:
        all_frames_data_list = []
    rgb_image = None
    for frame_index, jpg_file in enumerate(jpg_files):
        prev_rgb_image = rgb_image
        rgb_image = cv2.imread(jpg_file)
        if rgb_image is None:
            break
        rects = detect_balloons_in_frame(rgb_image, model, colors_ranges_info)
        rgb_image_with_tracking_data, rgb_image_with_id_data = ct_debugger.get_image_with_matching_objects(rects, rgb_image, prev_rgb_image, frame_index)
        save_frame(rgb_image_with_id_data, images_id_output_folder, frame_index)
        cv2.imshow("rgb_image_with_id_data", rgb_image_with_id_data)
        key = cv2.waitKey(frame_milliseconds) & 0xFF
        if key == ord("q"):
            break

        save_frame(rgb_image_with_tracking_data, images_tracking_output_folder, frame_index)
        cv2.imshow("rgb_image_with_tracking_data", rgb_image_with_tracking_data)
        key = cv2.waitKey(frame_milliseconds) & 0xFF
        if key == ord("q"):
            break


    cv2.destroyAllWindows()

    if debug_mode:
        pass
        #plot_explanation_for_centroid_tracker(all_frames_data_list)

    video_path_id_data = videos_id_output_folder + '/' + folder_name + 'tracking_balloons.avi'
    common_utils.create_video(frames_path=images_id_output_folder, frame_extension='jpg', video_path=video_path_id_data, frame_rate=10)

    video_path_tracking_data = videos_tracking_output_folder + '/' + folder_name + 'tracking_balloons.avi'
    common_utils.create_video(frames_path=images_tracking_output_folder, frame_extension='jpg', video_path=video_path_tracking_data, frame_rate=10)


main()
