#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64, UInt8
from geometry_msgs.msg import Twist
from project_2.msg import Lane, Timer
import message_filters

class ControlLane():
	def __init__(self):
		self.sub_center = message_filters.Subscriber('/detect/center', Lane)
		self.sub_timer = message_filters.Subscriber('/stop/timer', Timer)

		self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_center, self.sub_timer], 10, 10)

		self.sub_max_vel = rospy.Subscriber('/control/max_vel', Float64, self.cbGetMaxVel, queue_size = 1)
		self.pub_cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

		self.lastError = 0
		self.last_angular = 0		# > 0 Left, < 0 Right
		self.MAX_VEL = 0.09

		self.stopped = False	# Initially not stopping
		self.stopped_timer = 0	# Initially not stopping

		rospy.on_shutdown(self.fnShutDown)

	def cbGetMaxVel(self, max_vel_msg):
		self.MAX_VEL = max_vel_msg.data

	def stop(self):
		twist = Twist()
		twist.linear.x = 0
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		twist.angular.z = 0
		return twist

	def straight(self):
		twist = Twist()
		twist.linear.x = 0.2
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		twist.angular.z = 0
		return twist

	def cbFollowLane(self, lane, timer):
		time = timer.data
		if time > 0: 
		# if True:	# For Debugging
			self.pub_cmd_vel.publish(self.stop())
			self.stopped = True		
			return

		if self.stopped_timer > 0:
			print("Going Straight")
			self.pub_cmd_vel.publish(self.straight())
			self.stopped_timer -= 1
			return
		if self.stopped:	# Stopped at intersection but is ready to go
			print("Start Going Straight")
			self.stopped = False	# reset
			self.stopped_timer = 20	# move streight
			return

		center = lane.center
		lanes = lane.lanes
		error = center - 500

		Kp = 0.0006 # 0.005 # 0.0025
		Kd = 0.0012  # 0.009 # 0.007
		twist = Twist()

		angular_z = (Kp * error + Kd * (error - self.lastError))
		self.lastError = error
		
		twist.linear.x = min(self.MAX_VEL * (abs((1 - abs(error) / 500)) ** 2.2), 0.2)

			
		'''if lanes != 2:
			twist.linear.x = 0.1
			angular_z = -max(1.2*angular_z, -2.0) if angular_z < 0 else -min(1.2*angular_z, 2.0)
		'''
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0

		# print(self.last_angular)
		twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
		'''if lanes == 0:	# Turn from last turn
			twist.linear.x = 0
			twist.angular.z = self.last_angular'''
		self.last_angular = twist.angular.z
		self.pub_cmd_vel.publish(twist)

	def fnShutDown(self):
		rospy.loginfo("Shutting Down. TurtleBot Stopping")
		self.pub_cmd_vel.publish(self.stop())

	def main(self):
		rospy.loginfo("ControlNode :: Spinning")
		self.ts.registerCallback(self.cbFollowLane)
		rospy.spin()

if __name__ == '__main__':
	rospy.init_node('control_node')
	node = ControlLane()
	node.main()
