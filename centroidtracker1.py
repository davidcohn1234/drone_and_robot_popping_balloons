from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=150):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.num_of_unique_objects_in_video = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, single_object_data):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.num_of_unique_objects_in_video] = single_object_data
		self.disappeared[self.num_of_unique_objects_in_video] = 0
		self.num_of_unique_objects_in_video += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def get_objects_centroids(self, objects_list):
		num_of_objects_in_current_frame = len(objects_list)
		objectCentroids = np.zeros((num_of_objects_in_current_frame, 2), dtype="int")
		for i in range(0, num_of_objects_in_current_frame):
			single_object_data = objects_list[i]
			objectCentroids[i, :] = single_object_data['center_point']
		return objectCentroids

	def update(self, objects_data):
		rects = [single_object_data['bounding_box'] for single_object_data in objects_data]
		# check to see if the list of input bounding box rectangles
		# is empty
		num_of_objects_in_current_frame = len(objects_data)
		if num_of_objects_in_current_frame == 0:
			# loop over any existing tracked objects and mark them
			# as disappeared
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				# if we have reached a maximum number of consecutive
				# frames where a given object has been marked as
				# missing, deregister it
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			# return early as there are no centroids or tracking info
			# to update
			return self.objects

		# # initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((num_of_objects_in_current_frame, 2), dtype="int")

		# loop over the bounding box rectangles
		for (i, single_object_data) in enumerate(objects_data):
			inputCentroids[i, :] = single_object_data['center_point']

		# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, num_of_objects_in_current_frame):
				self.register(objects_data[i])

		# otherwise, are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objects_list = list(self.objects.values())
			objectCentroids = self.get_objects_centroids(objects_list)


			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid


			D = dist.cdist(objectCentroids, inputCentroids)

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
				current_input_centroid = objects_data[col]
				self.objects[objectID] = current_input_centroid
				self.disappeared[objectID] = 0

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

			# otherwise, if the number of input centroids is greater
			# than the number of existing object centroids we need to
			# register each new input centroid as a trackable object
			#else:
			for col in unusedCols:
				self.register(objects_data[col])

		# return the set of trackable objects
		return self.objects