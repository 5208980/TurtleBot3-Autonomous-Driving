#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist

class TurtleBotControlNode():
	def __init__(self):
		self.name = "TurtleBotControlNode ::"
		self.sub_lane = rospy.Subscriber('/lane_detect/action', Float64, self.cbFollowLane, queue_size = 1)
		# self.sub_max_vel = rospy.Subscriber('/control/max_vel', Float64, self.cbGetMaxVel, queue_size = 1)
		self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size = 1)

		self.lastError = 0
		self.MAX_VEL = 0.12

		rospy.on_shutdown(self.turtlebot_stop)
		self.prev_twist = 0 

	def cbGetMaxVel(self, max_vel_msg):
		self.MAX_VEL = max_vel_msg.data

	'''
	0 - Go Straight
	1 - Stop
	2 - Stop at intersection
	3 -	Turn Left
	4 - Turn Right
	5 - Spin
	'''
	def cbFollowLane(self, center_msg):
		print(center_msg.data)
		twist = Twist()
		if center_msg.data == 0:
			twist.linear.x = 0.0
			# twist.linear.x = 0.0
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = 0
			self.pub_cmd.publish(twist)
		elif center_msg.data == 1:
			twist.linear.x = 0
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = 0
			self.pub_cmd.publish(twist)
		elif center_msg.data == 2:
			twist.linear.x = 0
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = 0
			self.pub_cmd.publish(twist)
		elif center_msg.data == 3:	# Turn Left
			twist.linear.x = 0.00
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = 0.1
			self.pub_cmd.publish(twist) 
		elif center_msg.data == 4: # Turn Right
			twist.linear.x = 0.00
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = -0.0
			self.pub_cmd.publish(twist)
		else:
			twist.linear.x = 0.00
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = 0.0
			self.pub_cmd.publish(twist)
			# self.pub_cmd.publish(self.prev_twist)

		self.prev_twist = twist

	def turtlebot_stop(self):
		rospy.loginfo("%s: Shutting down. cmd_vel will be 0", self.name)

		twist = Twist()
		twist.linear.x = 0
		twist.linear.y = 0
		twist.linear.z = 0
		twist.angular.x = 0
		twist.angular.y = 0
		twist.angular.z = 0
		self.pub_cmd.publish(twist)

	def main(self):
		rospy.spin()

if __name__ == '__main__':
	rospy.init_node('control_node')
	node = TurtleBotControlNode()
	node.main()
