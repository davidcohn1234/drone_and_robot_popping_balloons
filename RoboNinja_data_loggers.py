'''
file RoboNinja_data_loggers.py
data robot loggers functions sub functions :
The supported frequencies are 1, 5, 10, 20, 30, and 50.
based on file programmer_and_data_logger_robot.py

20-23/09/22
 adding timeTag for each measurement record. 
    named 'measTimeTag' in file.
'''

# required imports
import sys
if sys.platform == "win32":
    from asyncio.windows_events import NULL
import time
# from turtle import speed
# import cv2
import pandas as pd

import robomaster
from robomaster import robot
##########################################################
##########################################################

recDataColumns = {'esc':[], 'att':[], 'imu':[], 'mode':[], 'pos':[], 
                    'status':[], 'velocity':[], 'bat':[], 'TOF':[], 'arm':[]}

recDataColumns['esc'] = ['measCounter', 'measTime', \
                'speed_0', 'speed_1', 'speed_2', 'speed_3',\
                'angle_0', 'angle_1', 'angle_2', 'angle_3',\
                'timestamp_0', 'timestamp_1', 'timestamp_2', 'timestamp_3',\
                'state_0', 'state_1', 'state_2', 'state_3'
                ]

recDataColumns['att'] = ['measCounter', 'measTime', \
                'yaw', 'pitch', 'roll'
                ]

recDataColumns['imu'] = ['measCounter', 'measTime', \
                'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'
                ]
                
recDataColumns['pos'] = ['measCounter', 'measTime', \
                'pos_x', 'pos_y', 'pos_z'
                ]
                
recDataColumns['status'] = ['measCounter', 'measTime', \
                'static_flag, up_hill, down_hill, on_slope, pick_up, slip_flag, \
                    impact_x, impact_y, impact_z, roll_over, hill_static'
                ]
                       
recDataColumns['bat'] = ['measCounter', 'measTime', \
                'percent'
                ]

recDataColumns['TOF'] = ['measCounter', 'measTime', \
                'range_4','range_1'
                ]

recDataColumns['arm'] = ['measCounter', 'measTime', \
                'arm_pos_x','arm_pos_y', 'gripper_mode'
                ]

recData_esc = pd.DataFrame(columns=recDataColumns['esc'])
recData_att = pd.DataFrame(columns=recDataColumns['att'])
recData_imu = pd.DataFrame(columns=recDataColumns['imu'])
recData_pos = pd.DataFrame(columns=recDataColumns['pos'])
recData_bat = pd.DataFrame(columns=recDataColumns['bat'])
recData_tof = pd.DataFrame(columns=recDataColumns['TOF'])
recData_arm = pd.DataFrame(columns=recDataColumns['arm'])
last_measures = { 'measCounter'	:0, 'measTimeTag'	:0,
                'pos_x':0,	'pos_y':0,	'pos_z':0,
                'yaw'	:0, 'pitch'	:0, 'roll'	:0,
                'arm_pos_x':0,'arm_pos_y':0, 'gripper_mode':0,
                'range_4':0,'range_1':0,
                'acc_x'	:0,'acc_y':0, 'acc_z' :0,'gyro_x':0, 'gyro_y':0, 'gyro_z':0,
                'speed_0' :0, 'speed_1' :0, 'speed_2' :0, 'speed_3' :0,
                'battery_percent':0,
                }
last_camera_img = {'img':0, 'counter':0, 'measTimeTag':0}
startingLogTimeStr = 0
startTimeTag = 0
# recData = pd.DataFrame(columns=recDataColumns)
##########################################################

globalTrialCounter=0

##########################################################
def get_last_measures():
    return last_measures
def get_last_measure_item(itemStr):
    return last_measures[itemStr]

def get_gripper_last_mode():
    return get_last_measure_item('gripper_mode')
def get_last_range():
    return (get_last_measure_item('range_4'),get_last_measure_item('range_1'))
    
def get_time_last_meas():
    return (get_last_measure_item('measCounter'),
            get_last_measure_item('measTimeTag'))
    
def get_esc_last_speeds():
    return (get_last_measure_item('speed_0'),
            get_last_measure_item('speed_1'),
            get_last_measure_item('speed_2'),
            get_last_measure_item('speed_3')
            )

def get_pose_last_meas():
    return (get_last_measure_item('pos_x'),
            get_last_measure_item('pos_y'),
            get_last_measure_item('pos_z'),
            get_last_measure_item('yaw'),
            get_last_measure_item('pitch'), 
            get_last_measure_item('roll')
            )
            
# def get_arm_last_meas():
def get_head_last_meas():
    return (get_last_measure_item('arm_pos_x'),
            get_last_measure_item('arm_pos_y'),
            get_gripper_last_mode(),
            get_last_range()
            )
    
''' camera handling'''
def get_camera_newest(robot):
    return last_camera_img

def sample_camera_img(robot):
    img = robot.camera.read_cv2_image(strategy="newest") # todo : verify simMode
    last_camera_img['img']     = img  # .copy() ??
    last_camera_img['counter'] = last_camera_img['counter'] + 1
    last_camera_img['measTimeTag'] = get_time_last_meas()

def get_robo_cam_res():
    return last_camera_img.shape[0], last_camera_img[1]  # w,h
##########################################################
def sub_data_handler_arm_pos_and_gripper_mode(sub_info):
    # print("Robotic Arm: pos x:{0}, pos y:{1}".format(pos_x, pos_y))    
    global recData_arm
    global globalTrialCounter
    global last_measures

    globalTrialCounter = globalTrialCounter+1
    ##
    pos_x, pos_y = sub_info
    # last_measures 'measCounter'  'measTimeTag'
    last_measures['arm_pos_x']=pos_x
    last_measures['arm_pos_y']=pos_y
    ##
    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                pos_x, pos_y, get_gripper_last_mode()
                                ]],
                                columns=recDataColumns['arm']) 
    recData_arm = pd.concat([recData_arm, newMeasRow])
    # print("len(recData) : ",len(recData))

def sub_data_handler_gripper_status(sub_info):
    global last_measures
    status = sub_info
    last_measures['gripper_mode'] = status
    # print("gripper status:{0}.".format(status))

def sub_info_handler_esc(esc_info):
    global recData_esc
    global globalTrialCounter #  nonlocal ?
    global last_measures
    globalTrialCounter = globalTrialCounter+1
    
    speed, angle, timestamp, state = esc_info
    # last_measures  'measCounter'	:0, 'measTimeTag'	:0,
    last_measures['speed_0'] = speed[0]
    last_measures['speed_1'] = speed[1]
    last_measures['speed_2'] = speed[2]
    last_measures['speed_3'] = speed[3]

    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                speed[0],speed[1],speed[2],speed[3], 
                                angle[0], angle[1], angle[2], angle[3], 
                                timestamp[0], timestamp[1], timestamp[2], timestamp[3], 
                                state[0],state[1],state[2],state[3]
                                ]],
                                columns=recDataColumns['esc']) 
    recData_esc = pd.concat([recData_esc, newMeasRow])
    # print("len(recData_esc) : ",len(recData_esc))

def sub_info_handler_batt_percent(batter_info, ep_robot):    
    global recData_bat
    global globalTrialCounter 
    global last_measures

    globalTrialCounter = globalTrialCounter+1

    percent = batter_info
    # last_measures  'measCounter'	:0, 'measTimeTag'	:0,
    last_measures['battery_percent'] = percent
    
    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                percent
                                ]],
                                columns=recDataColumns['bat']) 
    recData_bat = pd.concat([recData_bat, newMeasRow])
    # print("len(recData_bat) : ",len(recData_bat))

    if False:
        ep_led = ep_robot.led
        brightness = int(percent * 255 / 100)
        ep_led.set_led(comp="all", r=brightness, g=brightness, b=brightness)

def sub_info_handler_att(attitude_info):    
    global recData_att
    global globalTrialCounter #  nonlocal ?
    global last_measures

    globalTrialCounter = globalTrialCounter+1

    yaw, pitch, roll = attitude_info    
    # last_measures  'measCounter'	:0, 'measTimeTag'	:0,
    last_measures['yaw']    = yaw
    last_measures['pitch']  = pitch
    last_measures['roll']   = roll
    
    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                yaw, pitch, roll
                                ]],
                                columns=recDataColumns['att']) 
    recData_att = pd.concat([recData_att, newMeasRow])
    # print("len(recData_att) : ",len(recData_att))
    
def sub_info_handler_pos(position_info):    
    global recData_pos
    global globalTrialCounter
    global last_measures

    globalTrialCounter = globalTrialCounter+1

    x, y, z = position_info
    # last_measures  'measCounter'	:0, 'measTimeTag'	:0,
    last_measures['pos_x']  = x
    last_measures['pos_y']  = y
    last_measures['pos_z']  = z
    
    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                x,y,z     # cs on register can be 0,1
                                ]],
                                columns=recDataColumns['pos'])
    recData_pos = pd.concat([recData_pos, newMeasRow])
    # print("len(recData_pos) : ",len(recData_pos))

def sub_info_handler_imu(imu_info):
    acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z = imu_info
    
    global recData_imu
    global globalTrialCounter #  nonlocal ?
    globalTrialCounter = globalTrialCounter+1
    
    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z 
                                ]],
                                columns=recDataColumns['imu']) 
    recData_imu = pd.concat([recData_imu, newMeasRow])
    # print("len(recData_imu) : ",len(recData_imu))

def sub_data_handler_TOF(sub_info, ep_robot):
    global recData_tof
    global globalTrialCounter #  nonlocal ?
    global last_measures
    
    # print("tof1:{0}  tof2:{1}  tof3:{2}  tof4:{3}".format(distance[0], distance[1], distance[2], distance[3]))
    
    globalTrialCounter = globalTrialCounter+1

    distance = sub_info

    last_measures['range_4'] = distance[3]
    last_measures['range_1'] = distance[0]

    newMeasRow = pd.DataFrame([[ globalTrialCounter, get_measTimeTag(),
                                distance[3],distance[0]
                                ]],
                                columns=recDataColumns['TOF']) 
    recData_tof = pd.concat([recData_tof, newMeasRow])
    
    ################################################ move to class inside 
    if False:
        print("len(recData_tof) : ",len(recData_tof))

def sub_info_handler_main_status(status_info):
    static_flag, up_hill, down_hill, on_slope, pick_up, slip_flag, impact_x, impact_y, impact_z, \
    roll_over, hill_static = status_info

def sub_data_handler_adapter(sub_info):
    io_data, ad_data = sub_info
    print("io value: {0}, ad value: {1}".format(io_data, ad_data))

def sub_info_handler_GENERAL(sub_info):
    print("sub info: {0}".format(sub_info))

##########################################################

def init_robot_parts():
    '''
    relevant robot parts are
    ep_robot
    servo (arm operated by servo 1 and servo 2). status are angles[deg]
    gripper . status is angle[deg]
    esc
    camera
    infrared sensor (range, by [cm])
    led
    sensors adapter (operate ir flashing led?)
    '''
    # creation 
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    robot_version = ep_robot.get_version()
    print("Robot Version: {0}".format(robot_version))

    ep_chassis  = ep_robot.chassis

    ep_camera   = ep_robot.camera
    
    ep_led      = ep_robot.led  # doesnt require close() later
    
    ep_gripper  = ep_robot.gripper

    if True:
        ep_gripper.close(power=50)
        # time.sleep(1)
        # ep_gripper.pause()
    
    ep_arm      = ep_robot.robotic_arm
    
    ep_battery  = ep_robot.battery

    if True:
        
        ep_servo = ep_robot.servo

        # 舵机3 转到0度
        ep_servo.moveto(index=3, angle=90).wait_for_completed()

    if False:
        
        ep_sensor_adaptor = ep_robot.sensor_adaptor
        
        ep_sensor_adaptor.sub_adapter(freq=5, callback=sub_data_handler_adapter)

        # 获取传感器转接板adc值
        adc = ep_sensor_adaptor.get_adc(id=1, port=1)
        print("sensor adapter id1-port1 adc is {0}".format(adc))

        # 获取传感器转接板io电平
        io = ep_sensor_adaptor.get_io(id=1, port=1)
        print("sensor adapter id1-port1 io is {0}".format(io))

        # 获取传感器转接板io电平持续时间
        duration = ep_sensor_adaptor.get_pulse_period(id=1, port=1)
        print("sensor adapter id1-port1 duration is {0}ms".format(duration))

    ep_sensor = ep_robot.sensor

    return ep_robot, ep_chassis, ep_camera, \
            ep_led, ep_gripper, ep_arm, ep_battery, \
                ep_sensor, \
                robot_version


def close_all_robot(ep_robot, ep_chassis, ep_camera, ep_gripper):
    '''
    close all robot parts that are set in init_robot_parts()
    '''
    
    ep_gripper.open(power=50)
    time.sleep(1)
    ep_gripper.pause()

    ep_robot.close()

    return True

##########################################################
def study_track_move(ep_camera, ep_chassis):
    global desired_moves

    print("start move")
    # ep_chassis.move(x=0, y=0, z=-45).wait_for_completed()  #todo: _sub - show image stream while moving
    # ep_robot.play_sound(robot.SOUND_ID_SCANNING).wait_for_completed()
    # ep_robot.play_audio(filename="demo1.wav").wait_for_completed()
       
    if False:
        # 指定麦轮速度
        speed = 50
        slp = 1

        # 转动右前轮
        ep_chassis.drive_wheels(w1=speed, w2=0, w3=0, w4=0)
        time.sleep(slp)
        
        x_val = 0.5
        y_val = 0.3
        z_val = 30

        # 前进 3秒
        ep_chassis.drive_speed(x=x_val, y=0, z=0, timeout=5)
        time.sleep(3)
    else:
            
        slp = 1

        time.sleep(slp)       
        

    if False:
        for move in desired_moves:
            ep_chassis.move(x=move['x'], y=move['y'], z=move['z'], 
                            xy_speed=move['xyVel'], z_speed=move['zVel']).wait_for_completed() 
            # ep_arm.moveto
            # ep_gripper.
            time.sleep(waitSec)

    if False:        
            
        # ep_arm.moveto(x=0, y=0).wait_for_completed()
        ep_arm.moveto(x=60, y=50).wait_for_completed()
        ep_arm.move(x=20).wait_for_completed()
        ep_arm.move(x=-20).wait_for_completed()
        ep_arm.move(y=20).wait_for_completed()
        ep_arm.move(y=-20).wait_for_completed()
        ep_arm.moveto(x=190, y=120).wait_for_completed() # corner
        ep_arm.moveto(x=70, y=150).wait_for_completed()  # skweez inner
        ep_arm.moveto(x=210, y=40).wait_for_completed()  # straight forward
        # ep_arm.moveto(x=220, y=5).wait_for_completed()  # straight forward
        ep_arm.moveto(x=220, y=0).wait_for_completed()  # straight forward
        ep_arm.move(y=-40).wait_for_completed()
        ep_arm.move(y=40).wait_for_completed()

    print("end move")

    return True
##########################################################
def set_loggers(robot, pos_cs=0, commonSubFreq=10):
    
    set_TimeTag0()

    ep_sensor  = robot.sensor
    ep_chassis = robot.chassis
    # ep_battery = robot.
    ep_arm     = robot.robotic_arm
    ep_gripper = robot.gripper
    ep_camera  = robot.camera

    # ep_sensor.sub_distance(freq=commonSubFreq, callback=sub_data_handler_TOF, args=ep_chassis)  # otherwise without param names
    ep_sensor.sub_distance(commonSubFreq, sub_data_handler_TOF, robot)  # otherwise without param names
    # ep_battery.sub_battery_info(10, sub_info_handler_batt_percent, ep_robot)
    ep_arm.sub_position(freq=commonSubFreq, callback=sub_data_handler_arm_pos_and_gripper_mode)
    ep_gripper.sub_status(freq=commonSubFreq, callback=sub_data_handler_gripper_status)
    
    ep_chassis.sub_esc(freq=commonSubFreq, callback=sub_info_handler_esc)  
    ep_chassis.sub_position(freq=commonSubFreq, callback=sub_info_handler_pos, cs=pos_cs)  # cs 0 is the default
    ep_chassis.sub_attitude(freq=commonSubFreq, callback=sub_info_handler_att)
    ep_chassis.sub_imu(freq=commonSubFreq, callback=sub_info_handler_imu)
    
    ep_camera.start_video_stream(display=False)

def set_TimeTag0():
    global startTimeTag
    millisec = int(round(time.time() * 1000))
    startTimeTag = millisec 

def get_measTimeTag():    
    global last_measures
    global startTimeTag
    # ref from https://www.geeksforgeeks.org/get-current-date-and-time-using-python/ 
    millisec = int(round(time.time() * 1000))
    last_measures['measTimeTag'] = millisec - startTimeTag 
    return millisec

def setStartTimeTag():
    global startingLogTimeStr
    startingLogTimeStr = time.strftime("%Y%m%d_%H%M%S")
    return startingLogTimeStr

def add_timeTag_toFile(prefix, sufix, useStartTimeTag=True):
    global startingLogTimeStr
     
    if useStartTimeTag:
        if startingLogTimeStr==0:
            print("startingLogTimeStr is 0 , is now set by calling setStartTimeTag()")
            setStartTimeTag()
        timetagStr = str(startingLogTimeStr)
    else: # use CURRENT time string:
        timetagStr = time.strftime("%Y%m%d_%H%M%S")
    fileName = prefix + "_" + timetagStr + sufix
    return fileName

def unsub_loggers(robot, writeXLSXatFinsh = True):
    
    ep_sensor  = robot.sensor
    ep_chassis = robot.chassis
    ep_arm     = robot.robotic_arm
    ep_gripper = robot.gripper
    ep_camera  = robot.camera

    
    ep_camera.stop_video_stream()    
    ep_chassis.unsub_imu()
    ep_chassis.unsub_attitude()
    ep_chassis.unsub_position()
    ep_chassis.unsub_esc()
    ep_gripper.unsub_status()
    ep_arm.unsub_position()
    # ep_battery.unsub_battery_info()
    ep_sensor.unsub_distance()

    if writeXLSXatFinsh:
        # compound all dataframes and output to 1 excel file
        print("concatenating dataFrames..")
        recData = pd.concat([recData_esc, recData_pos, recData_att, 
                             recData_imu, recData_tof, recData_arm, 
                             recData_bat]
                             )
        print("sorting dataFrame..")
        # recData = recData.sort_values(by='measTimeTag')
        outFileName = add_timeTag_toFile("recData",".xlsx")
        recData.to_excel( outFileName )
        print("recorded data written to xlsx (named ", outFileName, ")")
##########################################################
if __name__ == '__main__':

    print("sys.platform : ", sys.platform)
    # pre definition
    if False:
        robomaster.enable_logging_to_file()

    ep_robot, ep_chassis, ep_camera, ep_led, \
        ep_gripper, ep_arm, ep_battery, ep_sensor, \
            robot_ver = init_robot_parts()    

    # subs
    # ep_sensor.sub_distance(freq=10, callback=sub_data_handler_TOF)
    ep_sensor.sub_distance(10, sub_data_handler_TOF, ep_chassis)
    ep_battery.sub_battery_info(10, sub_info_handler_batt_percent, ep_robot)
    if False:
        ep_arm.sub_position(freq=10, callback=sub_data_handler_arm_pos)
        ep_gripper.sub_status(freq=10, callback=sub_data_handler_gripper_status)

    ep_chassis.sub_esc(freq=10, callback=sub_info_handler_esc)  
    ep_chassis.sub_position(freq=10, callback=sub_info_handler_pos, cs=0)  # cs 0 is the default - relative to starting point _first_flag , 1 is relative to world? battery start ?
    ep_chassis.sub_attitude(freq=10, callback=sub_info_handler_att)
    ep_chassis.sub_imu(freq=10, callback=sub_info_handler_imu)
    if False:
        ep_chassis.sub_status(freq=50, callback=sub_info_handler_main_status)
   
    # action
    study_track_move(ep_camera, ep_chassis)
          
    # stop listening utils
    if False:
        ep_chassis.unsub_status()
    ep_chassis.unsub_imu()
    ep_chassis.unsub_attitude()
    ep_chassis.unsub_position()
    ep_chassis.unsub_esc()

    if False:
        ep_gripper.unsub_status()
        ep_arm.unsub_position()
    ep_battery.unsub_battery_info()
    ep_sensor.unsub_distance()

    close_all_robot(ep_robot, ep_chassis, ep_camera, ep_gripper)

    if False:
        # compound all dataframes and output to 1 excel file
        recData = pd.concat([recData_esc, recData_att, recData_imu, recData_pos, recData_bat, recData_tof])
        recData = recData.sort_values(by='measTimeTag')
        timetag = time.strftime("%Y%m%d_%H%M%S")
        recData.to_excel("recData_"+timetag+".xlsx")

