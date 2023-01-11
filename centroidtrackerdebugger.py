from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2
import common_utils
import random

class CentroidTrackerDebugger():
	def __init__(self, maxDisappeared=50):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.num_of_unique_objects_in_video = 0
		self.objects = OrderedDict()
		self.objects_rects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared
		self.random_colors = [(0, 0, 255),
							 (0, 255, 0),
							 (255, 0, 0),
							 (0, 255, 255),
							 (255, 0, 255),
							 (255, 255, 0),
							 (0, 0, 0),
							 (0, 0, 127),
							 (0, 127, 0),
							 (127, 0, 127),
							 (127, 127, 0),
							 (0, 127, 127),
							 (127, 127, 127)]

	def register(self, rect, centroid_pixel_point):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.num_of_unique_objects_in_video] = centroid_pixel_point
		self.objects_rects[self.num_of_unique_objects_in_video] = rect
		self.disappeared[self.num_of_unique_objects_in_video] = 0
		self.num_of_unique_objects_in_video += 1
		min_color = 0
		max_color = 255
		r = int(min_color + random.random() * (max_color - min_color))
		g = int(min_color + random.random() * (max_color - min_color))
		b = int(min_color + random.random() * (max_color - min_color))
		new_color = (r, g, b)
		self.random_colors.append(new_color)

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.objects_rects[objectID]
		del self.disappeared[objectID]

	def plot_bounding_box_with_id(self, rgb_image, bounding_box, object_id):
		rgb_image_with_data = rgb_image.copy()

		x_min = round(bounding_box[0])
		y_min = round(bounding_box[1])
		x_max = round(bounding_box[2])
		y_max = round(bounding_box[3])

		start_point = (int(x_min), int(y_min))
		end_point = (int(x_max), int(y_max))

		x_center = round(0.5 * (x_min + x_max))
		y_center = round(0.5 * (y_min + y_max))
		centroid = np.array((int(x_center), int(y_center)))
		# coordinateText = "({}, {})".format(x_center, y_center)
		# cv2.putText(rgb_image_with_data, coordinateText, (centroid[0], centroid[1] + 30),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
		text = "ID {}".format(object_id)
		current_color = self.random_colors[object_id]
		cv2.putText(rgb_image_with_data, text, (centroid[0] - 10, centroid[1] - 10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
		cv2.circle(rgb_image_with_data, (centroid[0], centroid[1]), 10, current_color, -1)
		cv2.rectangle(rgb_image_with_data, start_point, end_point, current_color, thickness=2)
		return rgb_image_with_data



	def plot_bounding_box(self, rgb_image, current_bounding_box, color, style, disappeard_count):
		rgb_image_with_data = rgb_image.copy()

		x_min = round(current_bounding_box[0])
		y_min = round(current_bounding_box[1])
		x_max = round(current_bounding_box[2])
		y_max = round(current_bounding_box[3])

		start_point = (int(x_min), int(y_min))
		end_point = (int(x_max), int(y_max))

		x_center = round(0.5 * (x_min + x_max))
		y_center = round(0.5 * (y_min + y_max))
		centroid = np.array((int(x_center), int(y_center)))
		# coordinateText = "({}, {})".format(x_center, y_center)
		# cv2.putText(rgb_image_with_data, coordinateText, (centroid[0], centroid[1] + 30),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
		if disappeard_count >= 0:
			disappeard_count_text = "disappeard = {}".format(disappeard_count)
			# cv2.putText(rgb_image_with_data, disappeard_count_text, (centroid[0], centroid[1] + 30),
			# 			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
		cv2.circle(rgb_image_with_data, (centroid[0], centroid[1]), 7, color, -1)
		common_utils.drawrect(rgb_image_with_data, start_point, end_point, color, thickness=2, style=style)
		#cv2.rectangle(rgb_image_with_data, start_point, end_point, color, thickness=2)
		return rgb_image_with_data

	def get_image_with_matching_objects(self, objects_data, rgb_image, prev_rgb_image, frame_index):
		# check to see if the list of input bounding box rectangles
		# is empty
		if prev_rgb_image is None:
			rgb_image_with_tracking_data = rgb_image.copy()
		else:
			ratio_image_1 = 0.5
			rgb_image_with_tracking_data = (ratio_image_1 * rgb_image + (1 - ratio_image_1) * prev_rgb_image).astype(np.uint8)
		rgb_image_with_id_data = rgb_image.copy()
		frame_index_str = "{}".format(frame_index)
		org = (50, 50)
		cv2.putText(rgb_image_with_tracking_data, frame_index_str, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
		cv2.putText(rgb_image_with_id_data, frame_index_str, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)

		rects_ordered_dict = OrderedDict()
		for index, rect in enumerate(rects):
			rects_ordered_dict[index] = rect

		current_frame_objects_color = (0, 255, 0)
		video_objects_color = (255, 0, 0)


		if len(rects) == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
				else:
					rgb_image_with_tracking_data = self.plot_bounding_box(rgb_image_with_tracking_data, self.objects_rects[objectID], color=video_objects_color, style='dashed', disappeard_count=self.disappeared[objectID])


			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# use the bounding box coordinates to derive the centroid
			cX = round((startX + endX) / 2.0)
			cY = round((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(rects[i], inputCentroids[i])
				rgb_image_with_tracking_data = self.plot_bounding_box(rgb_image_with_tracking_data, rects[i], color=current_frame_objects_color, style='dashed', disappeard_count=-1)
				rgb_image_with_id_data = self.plot_bounding_box_with_id(rgb_image_with_id_data, rects[i], object_id=i)

		# otherwise, are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			D_min_axis_1 = D.min(axis=1)
			rows = D_min_axis_1.argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			D_argmin_a_1 = D.argmin(axis=1)
			cols = D_argmin_a_1[rows]

			# in order to determine if we need to update, register,
			# or deregister an object we need to keep track of which
			# of the rows and column indexes we have already examined
			usedRows = set()
			usedCols = set()

			# loop over the combination of the (row, column) index
			# tuples
			for (row, col) in zip(rows, cols):
				# if we have already examined either the row or
				# column value before, ignore it
				# val
				if row in usedRows or col in usedCols:
					continue

				# otherwise, grab the object ID for the current row,
				# set its new centroid, and reset the disappeared
				# counter
				objectID = objectIDs[row]
				current_input_centroid = inputCentroids[col]
				start_point = self.objects[objectID]
				end_point = current_input_centroid
				self.objects[objectID] = current_input_centroid
				rect_last_position = self.objects_rects[objectID]
				self.objects_rects[objectID] = rects[col]
				self.disappeared[objectID] = 0

				rgb_image_with_id_data = self.plot_bounding_box_with_id(rgb_image_with_id_data, rects[col], object_id=objectID)
				rgb_image_with_tracking_data = self.plot_bounding_box(rgb_image_with_tracking_data, rect_last_position, color=video_objects_color, style='regular', disappeard_count=self.disappeared[objectID])
				rgb_image_with_tracking_data = self.plot_bounding_box(rgb_image_with_tracking_data, rects[col], color=current_frame_objects_color, style='regular', disappeard_count=-1)
				line_color = (0, 0, 255)
				line_thickness = 2
				cv2.line(rgb_image_with_tracking_data, start_point, end_point, line_color, line_thickness)


				# indicate that we have examined each of the row and
				# column indexes, respectively
				usedRows.add(row)
				usedCols.add(col)

			# compute both the row and column index we have NOT yet
			# examined
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# in the event that the number of object centroids is
			# equal or greater than the number of input centroids
			# we need to check and see if some of these objects have
			# potentially disappeared
			#if D.shape[0] >= D.shape[1]:
				# loop over the unused row indexes
			for row in unusedRows:
				# grab the object ID for the corresponding row
				# index and increment the disappeared counter
				objectID = objectIDs[row]
				self.disappeared[objectID] += 1

				# check to see if the number of consecutive
				# frames the object has been marked "disappeared"
				# for warrants deregistering the object
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)
				else:
					rgb_image_with_tracking_data = \
						self.plot_bounding_box(rgb_image_with_tracking_data, self.objects_rects[objectID], color=video_objects_color, style='dashed', disappeard_count=self.disappeared[objectID])


			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			#else:
			for col in unusedCols:
				self.register(rects[col], inputCentroids[col])

		# return the set of trackable objects
		return rgb_image_with_tracking_data, rgb_image_with_id_data
		#return self.objects