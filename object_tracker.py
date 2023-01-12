from centroidtracker1 import CentroidTracker
from centroidtrackerdebugger import CentroidTrackerDebugger
import cv2
import torch
import os
import common_utils
import glob
from yolo_object_detection import YoloObjectDetection

def save_frame(rgb_image_with_id_data, images_output_folder, frame_index):
    file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
    cv2.imwrite(file_full_path, rgb_image_with_id_data)

def main():
    # initialize our centroid tracker and frame dimensions
    ct = CentroidTracker()
    ct_debugger = CentroidTrackerDebugger()
    yolo_balloon_detection = YoloObjectDetection()

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

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
    model.conf = 0.8


    main_tracking_output_folder = './balloons_images_with_tracking_data'
    images_tracking_output_folder = main_tracking_output_folder + '/' + 'images'
    videos_tracking_output_folder = main_tracking_output_folder + '/' + 'videos'
    common_utils.create_empty_output_folder(images_tracking_output_folder)
    common_utils.create_empty_output_folder(videos_tracking_output_folder)

    main_id_output_folder = './balloons_images_with_id_data'
    images_id_output_folder = main_id_output_folder + '/' + 'images'
    videos_id_output_folder = main_id_output_folder + '/' + 'videos'
    common_utils.create_empty_output_folder(images_id_output_folder)
    common_utils.create_empty_output_folder(videos_id_output_folder)

    rgb_image = None
    for frame_index, jpg_file in enumerate(jpg_files):
        prev_rgb_image = rgb_image
        rgb_image = cv2.imread(jpg_file)
        if rgb_image is None:
            break
        objects_data = yolo_balloon_detection.detect_objects_in_frame(rgb_image, model)
        rgb_image_with_tracking_data, rgb_image_with_id_data = ct_debugger.get_image_with_matching_objects(objects_data, rgb_image, prev_rgb_image, frame_index)
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

    video_path_id_data = videos_id_output_folder + '/' + folder_name + 'tracking_balloons.avi'
    common_utils.create_video(frames_path=images_id_output_folder, frame_extension='jpg', video_path=video_path_id_data, frame_rate=10)

    video_path_tracking_data = videos_tracking_output_folder + '/' + folder_name + 'tracking_balloons.avi'
    common_utils.create_video(frames_path=images_tracking_output_folder, frame_extension='jpg', video_path=video_path_tracking_data, frame_rate=10)


main()
