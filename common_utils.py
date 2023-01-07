import os
import cv2
import shutil
import gdown
import glob
import os
import re
import platform
import numpy as np

platform_os = platform.system()
if platform_os == "Windows":
    import win32api


    def find_file(root_folder, rex):
        for root, dirs, files in os.walk(root_folder):
            for f in files:
                result = rex.search(f)
                if result:
                    first_path = os.path.join(root, f)
                    return first_path
        return None


    def find_file_in_all_drives(file_name):
        # create a regular expression for the file
        rex = re.compile(file_name)
        for drive in win32api.GetLogicalDriveStrings().split('\000')[:-1]:
            first_path = find_file(drive, rex)
            if first_path is not None:
                return first_path
        return None


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    if style=='regular':
        cv2.rectangle(img, pt1, pt2, color, thickness)
    else:
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        drawpoly(img, pts, color, thickness, style)


def create_video(frames_path, frame_extension, video_path, frame_rate):
    frames_files_full_paths = glob.glob(f'{frames_path}/*.{frame_extension}')
    frames_files_full_paths = sorted(frames_files_full_paths)
    first_frame_full_path = frames_files_full_paths[0]
    first_frame = cv2.imread(first_frame_full_path)
    first_frame_shape = first_frame.shape
    height = first_frame_shape[0]
    width = first_frame_shape[1]
    frameSize = (width, height)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, frameSize)
    num_of_frames = len(frames_files_full_paths)
    for idx, filename in enumerate(frames_files_full_paths):
        print(f'creating video: frame {idx + 1} out of {num_of_frames}')
        img = cv2.imread(filename)
        out.write(img)
    out.release()


def extract_frames_from_videos(videos_and_images_folder):
    video_folder_path = videos_and_images_folder + '/videos'
    images_folder_path = videos_and_images_folder + '/images'
    video_names = os.listdir(video_folder_path)
    video_names.sort()

    isFolderExist = os.path.exists(images_folder_path)
    if isFolderExist:
        return
    else:
        os.mkdir(images_folder_path)

    video_count = 0
    num_of_videos = len(video_names)
    for video_name_with_extension in video_names:
        video_count += 1
        video_name_without_extension = os.path.splitext(video_name_with_extension)[0]
        current_video_full_path = video_folder_path + '/' + video_name_with_extension
        vidcap = cv2.VideoCapture(current_video_full_path)
        current_video_num_of_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        success, image = vidcap.read()
        frame_count = 0
        images_sub_folder = images_folder_path + '/' + video_name_without_extension
        isSubFolderExist = os.path.exists(images_sub_folder)
        if isSubFolderExist == False:
            os.mkdir(images_sub_folder)
        while success:
            frame_count += 1
            frame_name = "%05d.jpg" % frame_count
            frame_full_path = images_sub_folder + '/' + frame_name
            cv2.imwrite(frame_full_path, image)  # save frame as JPEG file
            success, image = vidcap.read()
            print(
                f'video {video_count} out of {num_of_videos} videos. frame {frame_count} out of {current_video_num_of_frames} frames')
        print()


def copy_folders(input_main_folder_full_path, output_main_folder_full_path):
    sub_folders_names = os.listdir(input_main_folder_full_path)
    sub_folders_names.sort()
    sub_folder_counter = -1
    for single_sub_folder_name in sub_folders_names:
        sub_folder_counter += 1
        source_sub_folder_full_path = input_main_folder_full_path + '/' + single_sub_folder_name
        destination_sub_folder = output_main_folder_full_path + '/' + single_sub_folder_name
        is_subfolder_exist = os.path.exists(destination_sub_folder)
        if not is_subfolder_exist:
            os.makedirs(destination_sub_folder)
        files = [f.path for f in os.scandir(source_sub_folder_full_path) if f.is_file()]
        files.sort()
        counter = 0
        # counter = 1000 * sub_folder_counter
        for single_file in files:
            counter += 1
            destination_file_name = f"{counter:05}" + '.jpg'
            destination_file_full_path = destination_sub_folder + '/' + destination_file_name
            print(f'copying {single_file} to {destination_file_full_path}')
            shutil.copyfile(single_file, destination_file_full_path)


def download_input_images_from_google_drive(zip_folder, zip_file_id):
    input_data_folder_name = 'input_data'
    input_data_folder_full_path = zip_folder + '/' + input_data_folder_name
    is_folder_exist = os.path.exists(input_data_folder_full_path)
    if is_folder_exist:
        # if it already exists then return (nothing to do here).
        # otherwise, it will be created when unzipping the zip file
        return

    google_drive_prefix_url = 'https://drive.google.com/uc?id='

    zip_file_name = 'input_data.zip'
    print()

    url = google_drive_prefix_url + zip_file_id
    output_zip_file_full_path = zip_folder + '/' + zip_file_name
    is_zip_file_exist = os.path.exists(output_zip_file_full_path)
    if not is_zip_file_exist:
        print(f'Downloading {zip_file_name}')
        gdown.download(url, output_zip_file_full_path, quiet=False)
        print('Finished downloading zip file')
    else:
        print(f'{zip_file_name} already exists')
    print()
    print(f'Extracting {zip_file_name}')
    unzip_file(output_zip_file_full_path, zip_folder)
    print(f'Finished extracting {zip_file_name}')


def unzip_file(zip_file_full_path, extract_dir):
    archive_format = "zip"
    shutil.unpack_archive(zip_file_full_path, extract_dir, archive_format)
