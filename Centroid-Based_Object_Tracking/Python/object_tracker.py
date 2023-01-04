# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from centroidtracker1 import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import torch
import os


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


def main():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    (H, W) = (None, None)

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    # vs = VideoStream(src=0).start()

    input_file_name = 'balloons_video_ninja_room.mp4'
    #input_file_name = 'david.MOV'
    input_file_full_path = f'../../input_data/videos/{input_file_name}'
    vs = cv2.VideoCapture(input_file_full_path)
    # vs = cv2.VideoCapture(0)
    time.sleep(2.0)

    green_color_info = {"name": 'green', "lower": (42, 101, 96), "upper": (83, 168, 164), "rgb_color": (0, 255, 0)}
    red_color_info = {"name": 'red', "lower": (165, 132, 140), "upper": (203, 225, 222), "rgb_color": (0, 0, 255)}
    blue_color_info = {"name": 'blue', "lower": (89, 129, 123), "upper": (165, 241, 255), "rgb_color": (255, 0, 0)}
    orange_color_info = {"name": 'orange', "lower": (6, 58, 135), "upper": (26, 228, 255), "rgb_color": (0, 165, 255)}
    purple_color_info = {"name": 'purple', "lower": (81, 78, 106), "upper": (152, 160, 173), "rgb_color": (153, 51, 102)}
    colors_ranges_info = [green_color_info, red_color_info, blue_color_info, orange_color_info, purple_color_info]

    random_colors = [(0, 0, 255),
                     (0, 255, 0),
                     (255, 0, 0),
                     (0, 255, 255),
                     (255, 0, 255),
                     (255, 255, 0),
                     (0, 0, 0),
                     (0, 0, 127),
                     (0, 127, 0),
                     (127, 0, 127),
                     (127, 127, 0),
                     (0, 127, 127),
                     (127, 0, 127),
                     (127, 127, 0),
                     (127, 127, 127)]

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../balloons_weights.pt')
    model.conf = 0.8
    frame_index = 0

    images_output_folder = './balloons_images_with_data'

    isExist = os.path.exists(images_output_folder)
    if not isExist:
        os.makedirs(images_output_folder)

    # loop over the frames from the video stream
    while True:
        _, frame = vs.read()
        if frame is None:
            break
        frame_index += 1
        # if frame_index == 459:
        if frame_index == 217:
            david = 5
        rects = detect_balloons_in_frame(frame, model, colors_ranges_info)
        objects, mapping_object_ids_to_input_centroids = ct.update(rects)

        frame_with_ballons_data = frame.copy()
        frame_index_str = "{}".format(frame_index)
        org = (50, 50)
        cv2.putText(frame_with_ballons_data, frame_index_str, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
        num_of_balloons_for_current_frame = len(rects)
        for balloon_index in range(0, num_of_balloons_for_current_frame):
            current_bounding_box = rects[balloon_index]
            x_min = round(current_bounding_box[0])
            y_min = round(current_bounding_box[1])
            x_max = round(current_bounding_box[2])
            y_max = round(current_bounding_box[3])

            start_point = (int(x_min), int(y_min))
            end_point = (int(x_max), int(y_max))

            x_center = round(0.5 * (x_min + x_max))
            y_center = round(0.5 * (y_min + y_max))



            objectID = mapping_object_ids_to_input_centroids[balloon_index]
            centroid = np.array((int(x_center), int(y_center)))
            current_color = random_colors[objectID]
            text = "ID {}".format(objectID)
            cv2.putText(frame_with_ballons_data, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
            coordinateText = "({}, {})".format(x_center, y_center)
            cv2.putText(frame_with_ballons_data, coordinateText, (centroid[0] + 15, centroid[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
            cv2.circle(frame_with_ballons_data, (centroid[0], centroid[1]), 10, current_color, -1)
            cv2.rectangle(frame_with_ballons_data, start_point, end_point, current_color, thickness=2)


        # # loop over the tracked objects
        # for (objectID, centroid) in objects.items():
        #     # draw both the ID of the object and the centroid of the
        #     # object on the output frame
        #     current_color = random_colors[objectID]
        #     text = "ID {}".format(objectID+1)
        #     cv2.putText(frame_with_ballons_data, text, (centroid[0] - 10, centroid[1] - 10),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
        #     cv2.circle(frame_with_ballons_data, (centroid[0], centroid[1]), 10, current_color, -1)


        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
        cv2.imwrite(file_full_path, frame_with_ballons_data)
        # show the output frame
        cv2.imshow("frame_with_ballons_data", frame_with_ballons_data)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()


main()
