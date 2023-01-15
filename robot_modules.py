from robomaster import robot



class Robot:
    def __init__(self):
        self.ninja1 = self.initialize_robot()

    def initialize_robot(self):
        ninja1 = robot.Robot()
        ninja1.initialize(conn_type="ap")
        return ninja1


    def drive_speed(self, forward_backward, left_right, yaw_speed):
        self.ninja1.chassis.drive_speed(forward_backward, left_right, yaw_speed)
