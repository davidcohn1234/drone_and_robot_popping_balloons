import cv2
import torch
import common_utils
import numpy as np
from yolo_object_detection import YoloObjectDetection

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


def main():
    input_file_name = 'balloons_video_ninja_room.mp4'
    input_file_full_path = f'./input_data/videos/{input_file_name}'
    vid = cv2.VideoCapture(input_file_full_path)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
    model.conf = 0.8

    main_output_folder = './balloons_images_with_color_name'
    images_output_folder = main_output_folder + '/' + 'images'
    videos_output_folder = main_output_folder + '/' + 'videos'
    common_utils.create_empty_output_folder(images_output_folder)
    common_utils.create_empty_output_folder(videos_output_folder)

    yolo_balloon_detection = YoloObjectDetection()

    frame_index = 0
    while True:
        frame_index += 1
        _, rgb_image = vid.read()
        if rgb_image is None:
            break
        print(f'frame_index = {frame_index}')
        objects_data = yolo_balloon_detection.detect_objects_in_frame(rgb_image, model)
        rgb_image_with_ballons_data = yolo_balloon_detection.create_frame_with_objects_data(rgb_image, objects_data)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 255)
        thickness = 2
        org = (20, 50)
        cv2.putText(rgb_image_with_ballons_data, f'{frame_index}', org, font, fontScale, color, thickness, cv2.LINE_AA)

        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
        cv2.imwrite(file_full_path, rgb_image_with_ballons_data)

        cv2.imshow('rgb_image_with_ballons_data', rgb_image_with_ballons_data)
        #print(results.pandas().xyxy[0])
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

    video_path_balloons_with_colors = videos_output_folder + '/' + 'balloons_with_colors_names.avi'
    common_utils.create_video(frames_path=images_output_folder, frame_extension='jpg', video_path=video_path_balloons_with_colors, frame_rate=10)

main()