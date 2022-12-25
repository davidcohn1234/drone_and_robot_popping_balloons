from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=3):
		# initialize the next unique object ID along with two ordered
		# dictionaries used to keep track of mapping a given object
		# ID to its centroid and number of consecutive frames it has
		# been marked as "disappeared", respectively
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()

		# store the number of maximum consecutive frames a given
		# object is allowed to be marked as "disappeared" until we
		# need to deregister the object from tracking
		self.maxDisappeared = maxDisappeared

	def register(self, centroid):
		# when registering an object we use the next available object
		# ID to store the centroid
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):
		# to deregister an object ID we delete the object ID from
		# both of our respective dictionaries
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, cX, cY,balloon_color,radius,xoffset,yoffset,frame_id):
	# def update(self, rects):

		# check to see if the list of input bounding box rectangles
		# is empty
		# if len(rects) == 0:
		# 	# loop over any existing tracked objects and mark them
		# 	# as disappeared
		# 	for objectID in list(self.disappeared.keys()):
		# 		self.disappeared[objectID] += 1
		#
		# 		# if we have reached a maximum number of consecutive
		# 		# frames where a given object has been marked as
		# 		# missing, deregister it
		# 		if self.disappeared[objectID] > self.maxDisappeared:
		# 			self.deregister(objectID)
		#
		# 	# return early as there are no centroids or tracking info
		# 	# to update
		# 	return self.objects

		# initialize an array of input centroids for the current frame
		inputCentroids = np.zeros((len(cX), 7), dtype="int")

		# loop over the bounding box rectangles
		# for (i, (startX, startY, endX, endY)) in enumerate(rects):
		# 	# use the bounding box coordinates to derive the centroid
		# 	cX = int((startX + endX) / 2.0)
		# 	cY = int((startY + endY) / 2.0)
		# 	inputCentroids[i] = (cX, cY)
		for i in range(0,len(cX)):
			inputCentroids[i] = (cX[i], cY[i],radius[i],xoffset[i],yoffset[i],balloon_color[i],frame_id)

# if we are currently not tracking any objects take the input
		# centroids and register each of them
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])

		# otherwise, are are currently tracking objects so we need to
		# try to match the input centroids to existing object
		# centroids
		else:
			if inputCentroids.shape[0] == 0: #david - I have no idea if that's correct
				return self.objects

			# grab the set of object IDs and corresponding centroids
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			# compute the distance between each pair of object
			# centroids and input centroids, respectively -- our
			# goal will be to match an input centroid to an existing
			# object centroid
			np_objectCentroids = np.array(objectCentroids)

			# center_x1 = 0
			# center_y1 = 0
			# radius1 = 0
			# x_offset1 = 0
			# y_offset1 = 0
			# balloon_color1 = 0
			# frame_id1 = 0
			# np_objectCentroids_balloon1 = np.array((center_x1, center_y1, radius1, x_offset1, y_offset1, balloon_color1, frame_id1))
			# np_objectCentroids_balloon2 = np.array((center_x1 + 100, center_y1, radius1, x_offset1, y_offset1, balloon_color1, frame_id1))
			# np_objectCentroids_balloon3 = np.array((center_x1 + 200, center_y1, radius1, x_offset1, y_offset1, balloon_color1, frame_id1))
			# np_objectCentroids = np.array((np_objectCentroids_balloon1, np_objectCentroids_balloon2, np_objectCentroids_balloon3))
			#
			# inputCentroids_balloon1 = np_objectCentroids_balloon1 + np.array((0, 0, 10, 0, 0, 0, 0))
			# inputCentroids_balloon2 = np_objectCentroids_balloon2 + np.array((0, 0, 50, 0, 0, 0, 0))
			# inputCentroids_balloon3 = np_objectCentroids_balloon3 + np.array((0, 0, 150, 0, 0, 0, 0))

			center_x1 = 0
			center_y1 = 0
			np_objectCentroids_balloon1 = np.array((center_x1, center_y1))
			np_objectCentroids_balloon2 = np.array((center_x1 + 100, center_y1))
			np_objectCentroids_balloon3 = np.array((center_x1 + 200, center_y1))
			np_objectCentroids = np.array((np_objectCentroids_balloon1, np_objectCentroids_balloon2, np_objectCentroids_balloon3))

			inputCentroids_balloon1 = np_objectCentroids_balloon1 + np.array((0, 15))
			inputCentroids_balloon2 = np_objectCentroids_balloon2 + np.array((0, 5))
			inputCentroids_balloon3 = np_objectCentroids_balloon3 + np.array((0, 1225))
			inputCentroids_balloon4 = np_objectCentroids_balloon3 + np.array((0, 25))

			inputCentroids = np.array((inputCentroids_balloon1, inputCentroids_balloon2, inputCentroids_balloon3, inputCentroids_balloon4))

			D = dist.cdist(np_objectCentroids, inputCentroids)
			#D[i, j] = np.linalg.norm(np_objectCentroids[i, :] - inputCentroids[j, :])

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list

			# D_min_0[i] = min dist from inputCentroids[i, :] to any of the balloons in np_objectCentroids
			D_min_0 = D.min(axis=0)

			# D_min_1[i] = min dist from np_objectCentroids[i, :] to any of the balloons in inputCentroids
			D_min_1 = D.min(axis=1)

			rows = D_min_1.argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			D_argmin_1 = D.argmin(axis=1)
			cols = D_argmin_1[rows]
			# if for example cols = [0, 1, 3] it means that:
			# balloon 0 in inputCentroids is suitable to balloon 0 in np_objectCentroids
			# balloon 1 in inputCentroids is suitable to balloon 1 in np_objectCentroids
			# balloon 2 in inputCentroids is suitable to balloon 3 in np_objectCentroids

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
				self.objects[objectID] = inputCentroids[col]
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
			if D.shape[0] >= D.shape[1]:
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
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		# return the set of trackable objects
		return self.objects