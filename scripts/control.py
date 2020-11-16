#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64, UInt8
from geometry_msgs.msg import Twist
from project_2.msg import Lane, Timer
import message_filters

class ControlLane():
	def __init__(self):
		# self.sub_center = rospy.Subscriber('/detect/center', Lane, self.cbFollowLane, queue_size = 1)
		self.sub_center = message_filters.Subscriber('/detect/center', Lane)
		self.sub_timer = message_filters.Subscriber('/stop/timer', Timer)

		self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_center, self.sub_timer], 10, 10)

		self.sub_max_vel = rospy.Subscriber('/control/max_vel', Float64, self.cbGetMaxVel, queue_size = 1)
		self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

		self.lastError = 0
		self.MAX_VEL = 0.12
		print("HERE")

		rospy.on_shutdown(self.fnShutDown)

	def cbGetMaxVel(self, max_vel_msg):
		self.MAX_VEL = max_vel_msg.data

	def generate_stop_twist(self):
		twist = Twist()
		twist.linear.x = 0
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		twist.angular.z = 0
		return twist

	def cbFollowLane(self, lane, timer):
		time = timer.data
		# if time > 0: 
		if True:
			self.pub_cmd_vel.publish(self.generate_stop_twist())
			return

		center = lane.center
		error = center - 500

		Kp = 0.0025 # 0.005 # 0.0025
		Kd = 0.0007  # 0.009 # 0.007
		twist = Twist()
		lanes = lane.lanes

		angular_z = (Kp * error + Kd * (error - self.lastError))
		
		twist.linear.x = min(self.MAX_VEL * (abs((1 - abs(error) / 500)) ** 2.2), 0.2)
		if lanes != 2:
			twist.linear.x = 0.1
			angular_z = -max(1.2*angular_z, -2.0) if angular_z < 0 else -min(1.2*angular_z, 2.0)
		self.lastError = error
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
		self.pub_cmd_vel.publish(twist)

	def fnShutDown(self):
		rospy.loginfo("Shutting down. cmd_vel will be 0")
		self.pub_cmd_vel.publish(self.generate_stop_twist())

	def main(self):
		rospy.loginfo("ControlNode :: Spinning")
		self.ts.registerCallback(self.cbFollowLane)
		rospy.spin()

if __name__ == '__main__':
	rospy.init_node('control_node')
	node = ControlLane()
	node.main()
