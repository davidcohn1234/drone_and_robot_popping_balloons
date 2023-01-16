import cv2
import torch
import common_utils
import glob
from tracking_objects import Tracker
from yolo_object_detection import YoloObjectDetection

def get_first_frame_dimensions(jpg_files):
    first_jpg_file = jpg_files[0]
    first_rgb_image = cv2.imread(first_jpg_file)
    (height, width, channels) = first_rgb_image.shape
    return (height, width, channels)

def main():
    folder_name = '01_david_house'
    input_folder_full_path = f'./input_data/images/' + folder_name
    jpg_files = sorted(glob.glob(input_folder_full_path + '/*.jpg'))
    (image_height, image_width, image_channels) = get_first_frame_dimensions(jpg_files)

    #model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
    model = torch.hub.load('./yolov5', 'custom', './balloons_weights.pt', source='local')
    #model = torch.hub.load('yolov5s-cls.pt', 'custom', path='./balloons_kaggle.pt', source='local')
    model.conf = 0.8

    main_output_folder = './balloons_images_with_color_name'
    images_output_folder = main_output_folder + '/' + 'images'
    videos_output_folder = main_output_folder + '/' + 'videos'
    common_utils.create_empty_output_folder(images_output_folder)
    common_utils.create_empty_output_folder(videos_output_folder)


    object_tracker = Tracker(model, images_output_folder, image_height, image_width)


    while True:
        object_tracker.track()



    cv2.destroyAllWindows()

    video_path_balloons_with_colors = videos_output_folder + '/' + 'balloons_with_colors_names.avi'
    common_utils.create_video(frames_path=images_output_folder, frame_extension='jpg', video_path=video_path_balloons_with_colors, frame_rate=10)

main()