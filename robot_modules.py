from robomaster import robot
#from robomaster import camera


class RobotModules:
    def __init__(self):
        self.ninja1 = self.initialize_robot()
        #self.ninja1.camera.start_video_stream(display=True, resolution=camera.STREAM_360P)
        self.ninja1.camera.start_video_stream(display=False) #TODO - add the stop command

    def initialize_robot(self):
        ninja1 = robot.Robot()
        ninja1.initialize(conn_type="ap")
        return ninja1

    def drive_speed(self, forward_backward, left_right, yaw_speed):
        if forward_backward is None or left_right is None or yaw_speed is None:
            return
        print(f'self.ninja1.chassis.drive_speed({forward_backward}, {left_right}, {yaw_speed})')
        self.ninja1.chassis.drive_speed(forward_backward, left_right, yaw_speed)
