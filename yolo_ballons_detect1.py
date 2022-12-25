import cv2
import torch
import os
import numpy as np
from datetime import datetime
import math
'''
Nov 26 - fixing unknown color bug @ detect_balloon_data 
'''
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
cut_image = 1
save_image = 1
write_on_image=1


# blue_color_info = {"name": 'blue', "lower": (63, 123, 66), "upper": (156, 255, 255), "rgb_color": (255, 0, 0)} # B G R
# green_color_info = {"name": 'green', "lower": (39, 170, 64), "upper": (97, 255, 255), "rgb_color": (0, 255, 0)}
# red_color_info = {"name": 'red', "lower": (0, 150, 69) , "upper": (17, 255, 174), "rgb_color": (0, 0, 255)}
# yellow_color_info = {"name": 'yellow', "lower": (18, 189, 112), "upper": (48, 255, 255), "rgb_color": (0, 255, 255)}
# red_color_info2 = {"name": 'red', "lower": (161, 70, 69) , "upper": (255, 255, 255), "rgb_color": (0, 0, 255)}
blue_color_info = {"name": 'blue', "lower": (69 ,172, 66), "upper": (144, 255, 255),
                   "rgb_color": (255, 0, 0)}  # B G R # 69 172 66 144 255 255
#lower and upper for green are actually for yellow. it is temporary
green_color_info = {"name": 'green', "lower": (28, 172, 122), "upper": (48, 255, 255),
                    "rgb_color": (0, 255, 0)}  # 52 145 75 69 255 255
red_color_info = {"name": 'red', "lower": (0, 171, 52), "upper": (14, 255, 123),
                  "rgb_color": (0, 0, 255)}  # 0 150 075 11 255 167  / 0 171 52 014 255 123
#lower and upper for yellow are actually for green. it is temporary
yellow_color_info = {"name": 'yellow', "lower": (52, 145, 75), "upper": (69, 255, 255),
                     "rgb_color": (0, 255, 255)}  # 28 172 122 48 255 255
red_color_info2 = {"name": 'red', "lower": (138, 91, 74), "upper": (255, 255, 255),
                   "rgb_color": (0, 0, 255)}  # 138 91 74 255 255 255

colors_ranges_info = [yellow_color_info,green_color_info, red_color_info, blue_color_info,red_color_info2]
unknown_color_info = {"name": 'unknown', "lower": (0, 0, 0), "upper": (0, 0, 0), "rgb_color": (0, 0, 0)}


# input_file_name = 'balloons_video_ninja_room'
# input_file_full_path = f'./input_data_balloons/{input_file_name}.mp4'
#vid = cv2.VideoCapture("rtsp://192.168.1.28:8901/live")  # For streaming links
# vid = cv2.VideoCapture(input_file_full_path)
#vid = cv2.VideoCapture(0)
# vid.set(cv2.CAP_PROP_BUFFERSIZE, 2)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='./balloons_weights.pt')
# model = torch.hub.load('/home/lab_user/Ninja-2022_new/yolov5', 'custom', path='/home/lab_user/Ninja-2022_new/challenge_2_balloons/balloons_kaggle.pt', source='local')
#model = torch.hub.load('./', 'custom', path='./balloons_kaggle.pt', source='local')
model.conf = 0.6

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
    balloon_max_pixel=colors_num_of_pixels[max_index]
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
        balloon_max_pixel=0
        #return unknown_color_info , balloon_max_pixel
        return blue_color_info, balloon_max_pixel #TODO - return it to unknown
    return balloon_data,balloon_max_pixel


def run_script(frame,move_factor=0,frame_index=0):
#model.cuda(),
    # frame_index = 0
    # while True:
#     frame_index += 1
    # _, frame = vid.read()
    # if frame is None:
    #     break
    # current_time = now.strftime("%H_%M_%S")
    sub_image=[]
    [frame_height, frame_width, channels] = frame.shape
    results = model(frame)
    midx = int(frame_width / 2)
    midy = int(frame_height/2)
    x_0,y_0,r_0=midx,midy,0
    david = results.pandas()
    david1 = david.xyxy[0]
    david2 = david1.to_numpy()
    num_of_balloons = david2.shape[0]
    x_ratio = 0.1
    y_ratio = 0.1
    # print(f'frame_index = {frame_index}')
    # now = datetime.now()
    # now = now.strftime("%H_%M_%S_%f")
    images_output_folder = './balloons_images_with_data_1'
    if save_image:
        isExist = os.path.exists(images_output_folder)
        if not isExist:
            os.makedirs(images_output_folder)
            os.makedirs(images_output_folder + '_orig_frame_')

    # if frame_index == 261:
    #     david8 = 8
    frame_with_ballons_data = frame.copy()
    x_cent=[]
    y_cent=[]
    radius=[]
    max_pix=[]
    center_dist=[]
    Subimage_pos=[]
    center_ballon=[]
    dist=[]
    balloon_color_name=[]
    balloons_color=[]
    xoffset = []
    yoffset = []
    for balloon_index in range(0,num_of_balloons):
        mask_RGB = np.zeros([frame_height, frame_width, channels], dtype=frame.dtype)
        current_balloon_data = david2[balloon_index]
        x_min = current_balloon_data[0]
        y_min = current_balloon_data[1]
        x_max = current_balloon_data[2]
        y_max = current_balloon_data[3]
        # if cut_image and frame_index>2:
        #     Subimage_pos.append([x_min, x_max, y_min, y_max])
        # else:
        #     Subimage_pos.append([0, frame_width, 0, frame_height])
        start_point = (int(x_min), int(y_min))
        end_point = (int(x_max), int(y_max))
        balloon_width = x_max - x_min
        balloon_height = y_max - y_min
        radius.append(min([balloon_width,balloon_height])/2)
        x_start = int(x_min + x_ratio * balloon_width)
        x_end = int(x_min + (1 - x_ratio) * balloon_width)

        y_start = int(y_min + y_ratio * balloon_height)
        y_end = int(y_min + (1 - y_ratio) * balloon_height)

        x_middle = int(0.5 * (x_min + x_max))
        y_middle = int(0.5 * (y_min + y_max)+balloon_height*(move_factor))
        x_cent.append(x_middle)
        y_cent.append(y_middle)
        xoff_temp=int(x_middle-midx)
        yoff_temp=int(midy - y_middle)
        xoffset.append(xoff_temp)  # store the xoffset and yoffset for each iteration of the loop
        yoffset.append(yoff_temp)

        dist.append(math.sqrt(xoff_temp**2+yoff_temp**2))

        # if len(prev_center_ballon)>0:
        #     center_dist.append(math.sqrt((prev_center_ballon[0]-x_middle)**2+(prev_center_ballon[1]-y_middle)**2))


        sub_image = frame[y_start : y_end, x_start : x_end, :]
        mask_RGB[y_start : y_end, x_start : x_end, :] = sub_image

        # if frame_index == 260:
        #     cv2.imshow('mask_RGB', mask_RGB)
        #     key = cv2.waitKey(0) & 0xFF
        if write_on_image:
            thickness = 2
            frame_with_ballons_data=cv2.circle(frame_with_ballons_data, [int(x_cent[-1]),int(0.5 * (y_min + y_max))],int(radius[-1]), (255, 255, 255), thickness)
            # frame_with_ballons_data=cv2.circle(frame_with_ballons_data, [int(x_cent[-1]),int(0.5 * (y_min + y_max))],5, (0, 0, 255), -1)
            frame_with_ballons_data=cv2.circle(frame_with_ballons_data, [int(midx),int(midy)],5, (0, 255, 0), -1) #frame center
            frame_with_ballons_data = cv2.circle(frame_with_ballons_data, [int(x_middle), int(y_middle)], 5, (255, 0, 0), -1) #ballon center

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame_with_ballons_data, f'YOLO', (1200, 30), font,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # now = datetime.now()
        if save_image:
            try:
                current_time = current_time.strftime("%H_%M_%S")
            except:
                pass
            # file_full_path = "{}/yolo_{:05d}{:s}.jpg".format(images_output_folder, frame_index, 'frame_with_ballons_data_'+ str(balloon_index)+' from ' + str(num_of_balloons) + '@' + current_time)
            # cv2.imwrite(file_full_path, frame_with_ballons_data)


            # now = datetime.now()
            try:
                current_time = current_time.strftime("%H_%M_%S")
            except:
                pass
            # file_full_path = "{}/yolo_{:05d}{:s}.jpg".format(images_output_folder, frame_index, 'frame_with_ballons_data_'+ str(balloon_index)+' from ' + str(num_of_balloons) + '_sub_image@' + current_time)
            # cv2.imwrite(file_full_path, mask_RGB)

        confidence = current_balloon_data[4]
        obj_class = current_balloon_data[5]
        obj_name = current_balloon_data[6]
        david5 = 5

        try:
            balloon_data ,balloon_max_pixel = detect_balloon_data(mask_RGB, sub_image)
            if balloon_max_pixel==0:
                print('unknown')
            balloon_color_name_temp = balloon_data["name"]
            balloon_color = balloon_data["rgb_color"]
            max_pix.append(balloon_max_pixel)
            balloon_color_name.append(balloon_color_name_temp)

            if 'red' in balloon_color_name_temp:
                balloons_color.append(1)
                frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, (0, 0, 255), thickness)
                #frame = cv2.rectangle(frame, start_point, end_point, (0, 0, 255), thickness)
            elif 'blue' in balloon_color_name_temp:
                balloons_color.append(2)
                frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, (255, 0,0), thickness)
                #frame = cv2.rectangle(frame, start_point, end_point, (255, 0,0), thickness)
            elif 'green' in balloon_color_name_temp:
                balloons_color.append(3)
                frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, (0, 255,0), thickness)
                #frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), thickness)
            elif 'yellow' in balloon_color_name_temp:
                balloons_color.append(4)
                frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, (0, 255, 255), thickness)
                #frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 255), thickness)
            else:
                balloons_color.append(0)
                frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, (0, 0, 0), thickness)
                #frame = cv2.rectangle(frame, start_point, end_point, (0, 0, 0), thickness)

            if write_on_image:
                thickness = 2
                frame_with_ballons_data = cv2.rectangle(frame_with_ballons_data, start_point, end_point, balloon_color, thickness)
                # frame = cv2.rectangle(frame, start_point, end_point, balloon_color, thickness)

                font = cv2.FONT_HERSHEY_SIMPLEX
                balloonFontScale = 0.8
                balloonThickness = 2
                center = (x_middle, y_middle)
                cv2.putText(frame_with_ballons_data, f'{balloon_color_name_temp} , num pixel: {balloon_max_pixel}', center, font,
                            balloonFontScale, balloon_color, balloonThickness, cv2.LINE_AA)

                if 'yellow' in balloon_color_name_temp or 'unknown' in balloon_color_name_temp:
                    dist.pop(-1)
                    max_pix.pop(-1)
                    x_cent.pop(-1)
                    y_cent.pop(-1)
                    radius.pop(-1)
                    balloons_color.pop(-1)
                    xoffset.pop(-1)
                    yoffset.pop(-1)
                else:
                    pass
                    # frame = cv2.circle(frame, [int(x_cent[-1]), int(0.5 * (y_min + y_max))], int(radius[-1]),
                    #                    (255, 255, 255), thickness)
                    # frame = cv2.circle(frame, [int(x_cent[-1]), int(0.5 * (y_min + y_max))], 5, (0, 0, 255), -1)
                    # frame = cv2.circle(frame, [int(midx), int(midy)], 5, (0, 255, 0), -1)
                    # frame = cv2.circle(frame, [int(x_middle), int(y_middle)], 5, (255, 0, 0), -1)

        except:
            pass

    # if len(prev_center_ballon)>0:
    #     min_center = min(center_dist)
    #     min_index = center_dist.index(min_center)

    #cv2.imshow('Video Live IP cam', results.render()[0])
    if 0:
        if num_of_balloons>0:
            if len(dist) > 0:
                min_value = min(dist)
                min_index = dist.index(min_value)
                x_0 = x_cent[min_index]
                y_0 = y_cent[min_index]
                r_0 = radius[min_index]
                Subimage_pos = Subimage_pos[min_index]
            elif len(max_pix)>0:
                max_value = max(max_pix)
                max_index = max_pix.index(max_value)
                x_0=x_cent[max_index]
                y_0=y_cent[max_index]
                r_0=radius[max_index]
                Subimage_pos=Subimage_pos[max_index]
            else:
                x_0=x_cent[0]
                y_0=y_cent[0]
                r_0=radius[0]
                Subimage_pos=Subimage_pos[0]

            center_ballon=[x_0,y_0]
            xoffset = int(x_0 - midx)  # store the xoffset and yoffset for each iteration of the loop
            yoffset = int(midy - y_0)
        else:
            xoffset = int(-10000)  # store the xoffset and yoffset for each iteration of the loop
            yoffset = int(-10000)
        if write_on_image:
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            org = (20, 50)
            cv2.putText(frame_with_ballons_data, f' off cent: ({xoffset}, {yoffset}),radi: {round(r_0,2)},frame: {frame_index}', org, font, fontScale, color, thickness, cv2.LINE_AA)
            # now = datetime.now()

    if save_image:
        try:
            current_time = current_time.strftime("%H_%M_%S_%f")
        except:
            current_time=''
            pass

        file_full_path = "{}/{:05d}.jpg".format(images_output_folder + '_orig_frame_', frame_index)
        cv2.imwrite(file_full_path, frame)
        file_full_path = "{}/{:05d}.jpg".format(images_output_folder, frame_index)
        cv2.imwrite(file_full_path, frame_with_ballons_data)



    # cv2.imshow('frame', frame_with_ballons_data)
    # #print(results.pandas().xyxy[0])
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord('q'):
    #     break

    # vid.release()
    # cv2.destroyAllWindows()

    return x_cent,y_cent,balloons_color,radius,xoffset,yoffset, frame_with_ballons_data