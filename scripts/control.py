#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

class TurtleBotControlNode():
	def __init__(self):
		self.name = "TurtleBotControlNode ::"
		self.sub_lane = rospy.Subscriber('/lane_detect/center', Float64, self.cbFollowLane, queue_size = 1)
		# self.sub_max_vel = rospy.Subscriber('/control/max_vel', Float64, self.cbGetMaxVel, queue_size = 1)
		self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

		self.lastError = 0
		self.MAX_VEL = 0.12

		rospy.on_shutdown(self.turtlebot_stop)

	def cbGetMaxVel(self, max_vel_msg):
		self.MAX_VEL = max_vel_msg.data

	def cbFollowLane(self, center_msg):
		center = center_msg.data

		error = center - 500

		Kp = 0.0025
		Kd = 0.007

		angular_z = Kp * error + Kd * (error - self.lastError)
		self.lastError = error
		twist = Twist()
		# print((1 - abs(error) / 500) ** 2.2)
		# twist.linear.x = min(self.MAX_VEL * (abs((1 - abs(error) / 500)) ** 2.2), 0.2)
		twist.linear.x = 0
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		# twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
		twist.angular.z = 0
		self.pub_cmd_vel.publish(twist)

	def turtlebot_stop(self):
		rospy.loginfo("%s: Shutting down. cmd_vel will be 0", self.name)

		twist = Twist()
		twist.linear.x = 0
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		twist.angular.z = 0
		self.pub_cmd_vel.publish(twist)

	def main(self):
		rospy.spin()

if __name__ == '__main__':
	rospy.init_node('control_node')
	node = TurtleBotControlNode()
	node.main()
