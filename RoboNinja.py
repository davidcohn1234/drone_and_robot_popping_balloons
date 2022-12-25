'''
RoboNinja class
class includes
    robot init, and closing or robot parts
    enables consideration on connected parts such as robotic_arm (with arm + gripper + speaker)
    sub and unsub registers, for measurements
    show image from cam
    handling user keys for interaction
#todo:
get/set other speed values for all movements - by paramerts from class caller
include special movements such as simon45scan, back to yaw0 , etc.
options for 'p' (process):
 show on screen indications for image lines, obstacles identifications
bool flag for operating imshow(). in case called from class caller. + cam_meas() to return that frame
re-run (replay?) - from previous video file
add side window for drwaing manuever so far
version 001.5
update date
    23/10/22
        fix for stop move command
        change class defaults
    03/10/22
    open close gripper by delta time
        time.sleep(0.1) instead 1 sec to pause()
    stop action reinforced, by arm 0,0 command - fixxed again later on 23/10
    1.gripper smaller open/close steps by 'o','c' chars
    3.speeds scale factor by '+','-' is functional?
    questions left:
        -NONE-
    23/09
    add info onscreen on video frame
    add 'p' option for image processing on stream frame
    add stop condition by front range ('TOF')
notes:
 The video stream
OUT: H.264-encoded real-time video stream data
 with a resolution of 1280�720 and a refresh rate of 30 fps.
 The video stream data must be correctly decoded to
  display the video picture in real time.
https://robomaster-dev.readthedocs.io/en/latest/text_sdk/protocol_struct.html
check:
https://robomasterpy.nanmu.me/en/latest/quickstart.html
intro:
https://robomaster-dev.readthedocs.io/en/latest/python_sdk/beginner_ep.html
'''
import time
import cv2
from robomaster import robot

import RoboNinja_data_loggers as roboLog

# todo: move under robo class
allowRoobPrints = True  # prints to console
recordToAvi = True  # True   # for robo cam
recordFinalXls = True
orgRes = (1920, 1088)  # screen res. # manually tested and set from debug
showInfoLayer = True
processImage = 0  # 0 - default to None. other int - cyclic functionchange

# commandFactor = 1  # moved inside NinjaRobot class
defaultSpeed_ChasisMoveLR = 1  # [m/s]     , moveLeft, moveRight
defaultSpeed_ChasisMoveFB = 1  # [m/s]     , moveForward, moveBackward
defaultSpeed_ChasisRotate = 50  # [deg/sec] , rotateRight, rotateLeft
defaultPower_Gripper = 15  # [%]       , openGripper, closeGripper,
defaultSpeed_Arm = 8  # [mm]      , armUp, armDown, armFwd, armBackwrd

######################################
simonFloorMap = [
    {"frameIndex": "0", "frameName": "Robot_N", "screenXY": (550, 0), "takenImage": 0},
    {"frameIndex": "1", "frameName": "Robot_NE", "screenXY": (1090, 0), "takenImage": 0},
    {"frameIndex": "2", "frameName": "Robot_E", "screenXY": (1090, 260), "takenImage": 0},
    {"frameIndex": "3", "frameName": "Robot_SE", "screenXY": (1090, 540), "takenImage": 0},
    {"frameIndex": "4", "frameName": "Robot_S", "screenXY": (550, 540), "takenImage": 0},
    {"frameIndex": "5", "frameName": "Robot_SW", "screenXY": (0, 540), "takenImage": 0},
    {"frameIndex": "6", "frameName": "Robot_W", "screenXY": (0, 260), "takenImage": 0},
    {"frameIndex": "7", "frameName": "Robot_NW", "screenXY": (0, 0), "takenImage": 0},
]
######################################
'''
class for EP Robot , with local adjustments
'''


class NinjaRobot:  # (robot)

    ''' default user params '''
    robotID = 1
    simMode = False
    isArmAvailable = True  # True  # includes gripper, speaker, arm_x,y
    isCamAvailable = True  # does DJI's video camera is mounted and connected to the Robot ?
    showCamFrame = True

    includeRangeStopCondition = False  # todo: add alarm if data not present or dead
    RangeFwdStopLimit = 200  # [mm]   # 20cm from sensor install to rgipper end,
    #  when in middle position

    outputAviName = roboLog.add_timeTag_toFile("NinjaCam" + str(robotID), ".avi")
    outputAviRes = (640, 480)
    outputAviFPS = 20

    ''' local members init '''
    keyMappings = {}
    commandFactor = 1  # changes will increase/decrease move/rotate speeds. by multiply.
    fwdMoveDisable = False  # user or local??

    def __init__(self, init_args):  # todo: add simModeRequest from outside
        self.robot = robot.Robot()
        try:
            self.robot.initialize(conn_type="ap")
        except:
            print("robot init failed.  \n  setting simMode       = True")
            self.simMode = True
            # import sys
            # sys.exit(1)
            return  # todo: until simMode is taken care at all functions..

        if not self.isCamAvailable:
            self.showCamFrame = False

        self.initParts()

        self.subscribeLoggers()

        self.initKeyMappings()

        ''' start operating phase '''

        if recordToAvi:
            self.video_writer = cv2.VideoWriter(self.outputAviName, cv2.VideoWriter_fourcc(*'MJPG'),
                                                self.outputAviFPS, self.outputAviRes)

        # self.camera.start_video_stream(display=False)

    def initParts(self):

        self.version = self.robot.get_version()
        print("Robot Version: {0}".format(self.version))

        self.chassis = self.robot.chassis
        self.arm = self.robot.robotic_arm
        self.gripper = self.robot.gripper  # ep_gripper.pause() ?

        self.camera = self.robot.camera
        self.led = self.robot.led  # doesnt require close() later
        self.battery = self.robot.battery
        self.IRsensor = self.robot.sensor  # range / TOF

        self.gripper.close(power=50)
        time.sleep(1)
        self.gripper.pause()

        if False:
            ep_sensor_adaptor = self.robot.sensor_adaptor
            ep_sensor_adaptor.sub_adapter(freq=5, callback=sub_data_handler_adapter)

            adc = ep_sensor_adaptor.get_adc(id=1, port=1)
            print("sensor adapter id1-port1 adc is {0}".format(adc))

            io = ep_sensor_adaptor.get_io(id=1, port=1)
            print("sensor adapter id1-port1 io is {0}".format(io))

            duration = ep_sensor_adaptor.get_pulse_period(id=1, port=1)
            print("sensor adapter id1-port1 duration is {0}ms".format(duration))

        if True:
            percent = 0
            # percent = 99
            brightness = int(percent * 255 / 100)
            self.led.set_led(comp="all", r=brightness, g=brightness, b=brightness)

    def subscribeLoggers(self):
        roboLog.set_loggers(self.robot, pos_cs=0)  # 0-relative to run start, 1-relative to battery start

    def initKeyMappings(self):
        self.keyMappings = {
            'o': self.openGripper,
            'c': self.closeGripper,
            'x': self.rotateRight,
            'z': self.rotateLeft,
            'w': self.moveForward,  # change to up arrow ?
            's': self.moveBackward,  # change to down aroow.. ?
            'a': self.moveLeft,
            'd': self.moveRight,
            'm': self.stopMove,  # add ESC ?
            '1': self.armUp,  # +
            '2': self.armDown,  # -
            '3': self.armFwd,  # +
            '4': self.armBackwrd,  # -
            '+': self.scaleFactorUp,
            '-': self.scaleFactorDown,
            'r': self.scaleFactorReset,
            'j': self.special_move_scan45deg
        }
        print("Robot chars for keyMappings:")
        print(self.keyMappings.keys())
        # print(self.keyMappings)

    ###### class utils #####
    def isSimMode(self):
        return self.simMode

    def roboPrint(self, msg):
        if allowRoobPrints:
            print(msg)

    def get_robo_state(self):
        return roboLog.get_head_last_meas()

    def openGripper(self):
        self.roboPrint("open gripper")
        self.gripper.open(power=defaultPower_Gripper * self.commandFactor)
        time.sleep(0.05)
        self.gripper.pause()

    def closeGripper(self):
        self.roboPrint("close gripper")
        self.gripper.close(power=defaultPower_Gripper * self.commandFactor)
        time.sleep(0.05)
        self.gripper.pause()

    def rotateRight(self):
        self.roboPrint("rotate right")
        self.chassis.drive_speed(0, 0, defaultSpeed_ChasisRotate * self.commandFactor)

    def rotateRightByParam(self, user_z=-45, user_z_speed=100):
        self.roboPrint("rotate right, by param, -45, 100")
        # TODO: self.chassis.drive_speed(0, 0, defaultSpeed_ChasisRotate * self.commandFactor)
        # + TODO: add timeout for this motion (or just sleep??)
        self.chassis.move(x=0, y=0, z=-45, z_speed=100).wait_for_completed()

    def rotateLeft(self):
        self.roboPrint("rotate left")
        self.chassis.drive_speed(0, 0, -defaultSpeed_ChasisRotate * self.commandFactor)

    def move_any(self, x_speed, y_speed, z_speed):
        self.chassis.drive_speed(x_speed, y_speed, z_speed)

    def moveForward_with_speed(self, defaultSpeed_ChasisMoveFB):
        if self.fwdMoveDisable == False:
            self.roboPrint("move Forward")
            self.chassis.drive_speed(defaultSpeed_ChasisMoveFB * self.commandFactor, 0, 0)
        else:
            self.roboPrint("STOP Forward request because of close range")
            self.stopMove()

    def moveForward(self):
        if self.fwdMoveDisable == False:
            self.roboPrint("move Forward")
            self.chassis.drive_speed(defaultSpeed_ChasisMoveFB * self.commandFactor, 0, 0)
        else:
            self.roboPrint("STOP Forward request because of close range")
            self.stopMove()

    # def moveForwardByParam(self, user_x=0.1, user_y=0, user_z=0, user_z_speed=10):
    #     if self.fwdMoveDisable == False:
    #         self.roboPrint("move Forward")
    #         self.chassis.move(x=user_x, y=user_y, z=user_z, z_speed=user_z_speed, timeout=0.5).wait_for_completed()
    #     else:
    #         self.roboPrint("STOP Forward request because of close range")
    #         self.stopMove()
    def moveBackward(self):
        self.roboPrint("move Backward")
        self.chassis.drive_speed(-defaultSpeed_ChasisMoveFB * self.commandFactor, 0, 0)

    def moveLeft(self):
        self.roboPrint("move Left")
        self.chassis.drive_speed(0, -defaultSpeed_ChasisMoveLR * self.commandFactor, 0)

    def moveRight(self):
        self.roboPrint("move Right")
        self.chassis.drive_speed(0, defaultSpeed_ChasisMoveLR * self.commandFactor, 0)

    def stopMove(self):
        self.roboPrint("STOP!")
        # self.chassis.drive_speed(0, 0, 0)
        # if self.isArmAvailable:
        #     # self.arm.move(x=0).wait_for_completed()  # stop also to prevent drift?? , # timeout=? # problem - takes arm to 0 point
        #     armX = roboLog.get_head_last_meas()[0]
        #     armY = roboLog.get_head_last_meas()[1]
        #     self.arm.moveto(x=armX, y=armY).wait_for_completed()
        self.chassis.drive_wheels(0, 0, 0, 0)

    def armUpdoun(self, speed):
        self.arm.move(y=speed).wait_for_completed()

    def armUp(self):
        self.roboPrint("arm up")
        self.arm.move(y=defaultSpeed_Arm).wait_for_completed()

    def armDown(self):
        self.roboPrint("arm down")
        self.arm.move(y=-defaultSpeed_Arm).wait_for_completed()

    def armFwd(self):
        self.roboPrint("arm fwd")
        self.arm.move(x=defaultSpeed_Arm).wait_for_completed()

    def armBackwrd(self):
        self.roboPrint("arm backward")
        self.arm.move(x=-defaultSpeed_Arm).wait_for_completed()

    def scaleFactorUp(self):
        if self.commandFactor > 0.1 and self.commandFactor < 1.5:
            self.commandFactor = self.commandFactor + 0.1
        elif self.commandFactor < 0.15 and self.commandFactor > 0:
            self.commandFactor = self.commandFactor + 0.01

        print("scale factor 'commandFactor' change to : ", self.commandFactor)

    def scaleFactorDown(self):
        if self.commandFactor > 0.1:
            self.commandFactor = self.commandFactor - 0.1
        elif self.commandFactor < 0.15 and self.commandFactor > 0.01:
            self.commandFactor = self.commandFactor - 0.01

        print("scale factor 'commandFactor' change to : ", self.commandFactor)

    def scaleFactorReset(self):
        self.commandFactor = 1
        print("scale factor 'commandFactor' reset to : ", self.commandFactor)

    def check_range_condition(self):
        if self.includeRangeStopCondition:
            if roboLog.get_last_range() < self.RangeFwdStopLimit:  # [mm]
                print("TOF is less then 1 [m]")
                self.fwdMoveDisable = True
                # self.moveForward()             # will stop the movement
                self.roboPrint("STOP Forward request because of close range")
                self.stopMove()
                print("maneuver is stoppd. TOF is less then 1 [m]")
            else:
                if self.fwdMoveDisable == True:
                    print("TOF is back to more then 1 [m]")
                self.fwdMoveDisable = False

    ############################################################
    def special_move_scan45deg_station45deg_func(self, ndx):
        global simonFloorMap

        # take image
        cv2.waitKey(1)
        time.sleep(1)
        cam_img = self.camera.read_cv2_image(strategy="newest")
        simonFloorMap[ndx]["takenImage"] = cam_img.copy()  # keep image copy, full res
        imgSmall = cv2.resize(simonFloorMap[ndx]["takenImage"], (300, 200))

        # display image on screen 25 squere map (5x5)

        if True:
            cv2.imshow(simonFloorMap[ndx]["frameName"], imgSmall)
            cv2.moveWindow(simonFloorMap[ndx]["frameName"],
                           simonFloorMap[ndx]["screenXY"][0],
                           simonFloorMap[ndx]["screenXY"][1])

            # write to file. full res image
        cv2.imwrite(simonFloorMap[ndx]["frameIndex"] + "_" +
                    simonFloorMap[ndx]["frameName"] + '.png',
                    simonFloorMap[ndx]["takenImage"])  # record section scan

    def special_move_scan45deg(self):  # was scan_45_simon_floor()
        global simonFloorMap

        for ndx in range(len(simonFloorMap)):
            self.moveForward()
            time.sleep(0.5)
            self.stopMove()

            print("take image")
            self.special_move_scan45deg_station45deg_func(ndx)
            print("finish display image")

            self.moveBackward()
            time.sleep(0.5)
            self.stopMove()

            print("start move")
            self.rotateRightByParam()
            print("end move")

            print("image file saved")

        # ? closing program ?? return True

    ############################################################
    def special_move_at_incline(self):
        state = True
        start_pitch = roboLog.get_pose_last_meas()[4]
        while state:
            pitch = roboLog.get_pose_last_meas()[4]
            print('pitch' + str(pitch))
            time.sleep(0.05)
            if (pitch - start_pitch) > 5:
                while pitch > 5:
                    print('moving up @ ' + str(pitch))
                    self.move_any(1, 0, 0)
                    pitch = roboLog.get_pose_last_meas()[4]
                    time.sleep(0.1)
                self.stopMove()
            elif (pitch - start_pitch) < -5:
                while pitch < -5:
                    print('moving down @ ' + str(pitch))
                    pitch = roboLog.get_pose_last_meas()[4]
                    time.sleep(0.1)
                self.stopMove()


if __name__ == '__main__':

    ninja1 = NinjaRobot('')

    if ninja1.isSimMode():
        print("what to do in sim mode of Robomaster EP ?? ")

    quitFlag = False
    while not quitFlag:

        # stream image from robot
        if False:
            img = ninja1.camera.read_cv2_image(strategy="newest")
        else:
            roboLog.sample_camera_img(ninja1)
            imgDict = roboLog.get_camera_newest(ninja1)
            img = imgDict['img']
            imgNdx = imgDict['counter']
        img = cv2.resize(img, ninja1.outputAviRes)
        if processImage == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pass
        elif processImage == 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            processImage = 0

        ninja1.check_range_condition()

        if showInfoLayer:
            dataStr = "img ndx: " + str(imgNdx)
            img = cv2.putText(img, dataStr, (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)  # ??
            timeSinceStart = roboLog.get_time_last_meas()[1] - 0  # todo: show X.# res.
            dataStr = "meas time: " + str(timeSinceStart)
            img = cv2.putText(img, dataStr, (320, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)  # ??

            if ninja1.isArmAvailable:
                # text_image = cv2.putText(img, 'Miracles of OpenCV', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),3)
                dataStr = "armX [mm]: " + str(roboLog.get_head_last_meas()[0])
                img = cv2.putText(img, dataStr, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                dataStr = "armY [mm]: " + str(roboLog.get_head_last_meas()[1])
                img = cv2.putText(img, dataStr, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                # dataStr = "gripper : " + str(roboLog.get_head_last_meas()[2])
                # img = cv2.putText(img, dataStr, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),2)

            # todo: put pos x,y of robot + add battery info

            dataStr = "forward range [mm]: " + str(roboLog.get_head_last_meas()[3])

            if ninja1.fwdMoveDisable == False:
                infoColor = (255, 255, 0)
            else:
                infoColor = (0, 0, 255)
            img = cv2.putText(img, dataStr, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, infoColor, 2)

            dataStr = "yaw [deg]: " + str(roboLog.get_pose_last_meas()[3])
            img = cv2.putText(img, dataStr, (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            dataStr = "Pitch [deg]: " + str(roboLog.get_pose_last_meas()[4])
            img = cv2.putText(img, dataStr, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("ninja1 Robot", img)

        if recordToAvi:
            ninja1.video_writer.write(img)

        key = cv2.waitKey(1) & 0xFF

        # ninjaRobot handle_cv2_keys(key)
        # KEYBOARD INTERACTIONS
        if key == ord('q'):
            quitFlag = True
            break

        elif key == ord('p'):  # convert to grayscale
            # processImage = not processImage
            processImage = processImage + 1
            if processImage > 5:
                processImage = 0
            print("processImage is now ", processImage)
        elif key == ord('i'):  # info - layer on/off write some text and save it
            showInfoLayer = not showInfoLayer
            print("showInfoLayer is now ", showInfoLayer)

        elif chr(key) in ninja1.keyMappings.keys():
            ninja1.keyMappings[chr(key)]()

            ##
    if recordToAvi:
        ninja1.video_writer.release()
        print(" AVI file is now closed ")
    roboLog.unsub_loggers(ninja1.robot, writeXLSXatFinsh=recordFinalXls)
    ##

    # ninja1.camera.stop_video_stream()
    ninja1.robot.close()

    cv2.destroyAllWindows()