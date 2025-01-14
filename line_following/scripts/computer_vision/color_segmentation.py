import cv2
import numpy as np
import pdb
from collections import defaultdict
#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################

def image_print(img, image_name="image"):
	"""
	Helper function to print out images, for debugging. Pass them in as a list.
	Press any key to continue.
	"""
	cv2.imshow(image_name, img)
	cv2.waitKey(0)
	# cv2.destroyAllWindows()

def cd_color_segmentation(img, template=None):
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; List of two tuples: (low values, high values) in where each value is (hue, sat, val)
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	x = y = w = h = 0
	# image_print(img) #prints image
	# bounding values: change for different lighting conditions in HSV
	if template is not None:
		lower_bounds = template[0]
		upper_bounds = template[1]
	else:
		lower_bounds = (10, 10, 120)
		upper_bounds = (30,255, 255)
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #converts bgr to hsv
	mask = cv2.inRange(hsv_img, lower_bounds, upper_bounds) #creates binary image with 1 = within bounds given
	element = np.ones((5, 5), np.uint8) #kernel for erosion/dilation
	erosion_dst = cv2.erode(mask, element, iterations=1) #shrinks mask area down
	mask = cv2.dilate(erosion_dst, element, iterations=1) #increases mask area
	# image_print(mask)
	hsv, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #finding areas were masks exist
	cnt = None
	max_area = -1
	# finding largest contour in mask
	for i in contours:
		area = i.shape[0]
		if area > max_area:
			cnt = i
			max_area = area
	if cnt is not None:
		x, y, w, h = cv2.boundingRect(cnt) # creates rectangle around largest contour
		cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0), 2) #adds box onto image
	#image_print(img)
	bounding_box = ((x,y),(x+w,y+h))

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box


def isolate_orange(img):
	'''
	Isolate orange colors from the image!
	'''
	image_print(img)

	LOWER_HUE_THRESH = 24.0 / 360.0
	UPPER_HUE_THRESH = 28.0 / 360.0
	LOWER_SAT_THRESH = 0.95 # percent
	LOWER_VAL_THRESH = 0.95 # percent

	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	img_shape = hsv_img.shape

	# Color Filter
	lower_hue_mask = hsv_img[:, :, 0] > LOWER_HUE_THRESH
	upper_hue_mask = hsv_img[:, :, 0] > UPPER_HUE_THRESH
	saturation_mask = hsv_img[:, :, 1] > LOWER_SAT_THRESH
	value_mask = hsv_img[:, :, 1] > LOWER_VAL_THRESH

	hue_mask = lower_hue_mask * upper_hue_mask
	test_img = np.dstack((hsv_img[:, :, 0]*hue_mask, hsv_img[:, :, 1]*hue_mask, hsv_img[:, :, 2]*hue_mask))
	
	image_print(cv2.cvtColor(test_img, cv2.COLOR_HSV2BGR))


	color_mask = lower_hue_mask * upper_hue_mask * saturation_mask * value_mask

	orange_filterd = np.dstack((hsv_img[:, :, 0] * color_mask,
								hsv_img[:, :, 1] * color_mask,
								hsv_img[:, :, 2] * color_mask))

	bgr_filtered = cv2.cvtColor(orange_filterd, cv2.COLOR_HSV2BGR)

	image_print(bgr_filtered)


def segment_angle_kmeans(lines, k=2, **kwargs):
	## https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
	"""Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle 
    to segment `k` angles inside `lines`.
    """

	# Define criteria = (type, max_iter, epsilon)
	default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
	criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
	flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
	attempts = kwargs.get('attempts', 10)

	# returns angles in [0, pi] in radians
	angles = np.array([line[0][1] for line in lines])
	# multiply the angles by two and find coordinates of that angle
	pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
					for angle in angles], dtype=np.float32)

	# run kmeans on the coords
	labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
	labels = labels.reshape(-1)  # transpose to row vec

	# segment lines based on their kmeans label
	segmented = defaultdict(list)
	for i, line in enumerate(lines):
		segmented[labels[i]].append(line)
	segmented = list(segmented.values())
	return segmented

def intersection(line1, line2, image_shape=None):
	# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])

	# return np.linalg.lstsq(A, b)[0]

    x0, y0 = np.linalg.lstsq(A, b, rcond=-1)[0]
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines, image_shape=None):
	# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
	# """Finds the intersections between groups of lines."""

	intersections = []
	for i, group in enumerate(lines[:-1]):
		for next_group in lines[i+1:]:
			for line1 in group:
				for line2 in next_group:
					found_intersection = intersection(line1, line2)

					if image_shape is not None:
						_x, _y = found_intersection[0]

						if 0 <= _x < image_shape[1] and 0 <= _y < image_shape[0]:
							intersections.append(found_intersection)
					else:
						intersections.append(found_intersection)
	return intersections


def lf_color_segmentation(img, template=None, pct=0.6, similarity_margin=0.2, horizontal_angle_margin=5*np.pi/180, visualize=False): #pct specifies which portion of the image to look in
	"""
	Implement orange line detection using color masking and hough transforms
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; List of two tuples: (low values, high values) in where each value is (hue, sat, val)
		pct: float; specifies which portion of the image to look in for the line (starting at the bottom)
		similarity_margin: difference in angles (rad) we want to use as similarity cutoff for a single line
		horizontal_angle_margin: +/- angle (rad) from horizontal that we consider as hard turn 
	Return:
		trajectory: list of (u, v); list of points on the orange line, starting from the bottom, unit in px

		returns None if no trajectory was found
		returns True if we should turn hard left
		retrusn False if we should turn hard right
	"""
	########## YOUR CODE STARTS HERE ##########
	horizontal_line_step = 15
	pixel_cutoff = pct
	# horizontal_angle_margin = 20 * np.pi / 180


	## ORANGE MASK
	orange_lower_bounds = (10, 0, 120)
	orange_upper_bounds = (30,255, 255)

	## BROWN MASK
	brown_lower_bounds = (10, 0, 0)
	brown_upper_bounds = (50,255, 120)

	# image_print(img) #prints image
	# bounding values: change for different lighting conditions in HSV
	if template is not None:
		lower_bounds = template[0]
		upper_bounds = template[1]
	else:
		lower_bounds = (5, 190, 100)
		upper_bounds = (40, 255, 255)
		
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #converts bgr to hsv
	img_shape = hsv_img.shape

	if visualize: print("Image shape: "+str(img_shape))

	start_x = int(np.floor(img_shape[0]*(1-1)))
	hsv_img_cropped = hsv_img[start_x:,:]
	mask = cv2.inRange(hsv_img_cropped, lower_bounds, upper_bounds) #creates binary image with 1 = within bounds given
	element = np.ones((5, 5), np.uint8) #kernel for erosion/dilation
	erosion_dst = cv2.erode(mask, element, iterations=1) #shrinks mask area down
	mask = cv2.dilate(erosion_dst, element, iterations=1) #increases mask area

	if visualize: image_print(mask, "mask")
	if visualize: print("First mask shape: "+str(mask.shape))

	### Now that we have a mask, do line detection
	canny_img = cv2.Canny(mask, 100, 200)
        
	big_element = np.ones((21, 21), np.uint8) #kernel for erosion/dilation
	mask = cv2.dilate(erosion_dst, big_element, iterations=1) #increases mask area

	if visualize: print("Second mask shape: "+str(mask.shape))

	# HoughLines(img, rho, theta, threshold, lines,)
	lines = cv2.HoughLines(canny_img, 1, np.pi/180, 60, None, 0, 0)

	if lines is not None and lines.size > 0:
		# Create three groups of lines (one of which should be completely horizontal)
		segmented = segment_angle_kmeans(lines, k=2)

		# sorted by theta's distance from pi/2, reversed so we look at more vertical lines first
		segmented = sorted(segmented, key=lambda g: (g[0][0][1]-np.pi/2)**2, reverse=True) 

		
		# Add horizontal lines to the image after segmentation! [r = y-int, theta = pi/2]
		horiz_lines = np.array([[img_shape[0], np.pi/2]])[np.newaxis, :, :]
		for intercept in range(horizontal_line_step, int(img_shape[0] * pixel_cutoff), horizontal_line_step):
			# y = 0 is at the top
			horiz_lines = np.vstack((horiz_lines, np.array([[img_shape[0] - intercept, np.pi/2]])[np.newaxis, :, :]))

		segmented.append(horiz_lines.tolist())

		# for some reason, we didn't get three line clusters?
		if len(segmented) != 3: return None

		blue_angles = []

		# add the different groups by color to the base image
		#				[RED, 		 BLUE, 		  BLACK]
		group_colors = [(0, 0, 255), (255, 0, 0), (25, 25, 25)]

		average_sin_angles = list()

		for g_idx, group in enumerate(segmented):
			# print("Group "+str(g_idx)+" has angle: "+str(group[0][0][1]))
			g_sin_angles = list()
			for l_idx, l in enumerate(group):
				rho = l[0][0]
				theta = l[0][1]
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a * rho
				y0 = b * rho

				g_sin_angles.append(theta)
				if (g_idx == 1): blue_angles.append(theta)

				pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
				pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
				
				# UNCOMMENT THE LINE BELOW TO SEE THE LINES ON THE IMAGE!
				if visualize: cv2.line(img, pt1, pt2, group_colors[g_idx], 2, cv2.LINE_AA)

			average_sin_angles.append(np.average(np.array(g_sin_angles)))

		if visualize: print(average_sin_angles)
		# are the two classes of lines close in angle?
		if np.abs(average_sin_angles[0] - average_sin_angles[1]) < similarity_margin:
			detected_single_lane = True
		else:
			detected_single_lane = False


		blue_sqr_dst_from_horizontal = (np.array(blue_angles) - np.pi/2)**2
		max_blue_dist_from_horizontal = np.sqrt(np.max(blue_sqr_dst_from_horizontal))
		
		# image_print(img, "test")

		# first two line groupings are theoretically the lines we care about
		# the third line group should be the horizontal group
		main_intersections = np.array(segmented_intersections(segmented[:2], mask.shape)) # intersect first two
		lower_intersections = np.array(segmented_intersections([segmented[0], segmented[2]], mask.shape)) # intersect first with horizontal
		upper_intersections = np.array(segmented_intersections(segmented[1:], mask.shape)) # intersect second with horizontal

		# average point of the two main classes intersecting, used to tell us where to switch line classes
		average_point = np.rint(np.average(main_intersections, axis=0)).astype(np.int32)
		# cv2.circle(img, tuple(average_point[0].tolist()), 5, (255, 255, 0), 1)

		# find the points intersecting with the lower classification and the horizontal lines
		filter_lower = lower_intersections[:,:,1] > average_point[0][1]

		filter_lower = lower_intersections[:,:,1] > -10 # should always nonneg
		lower_intersections_to_use = lower_intersections[filter_lower, :]
		# draw them (but not anymore)
		# for idx, pt in enumerate(lower_intersections_to_use): cv2.circle(img, tuple(pt), 5, (0, 255, 255), 1)

		# are we making a 90 degree turn?
		if max_blue_dist_from_horizontal <= horizontal_angle_margin:
			# yes, get the self intersections on the horizontal track and determine the point mass
			second_self_intersections = segmented_intersections([segmented[1], segmented[1]], mask.shape)

			use_points = list()
			for pt in second_self_intersections:
				if mask[pt[0][1], pt[0][0]] > 200:
					use_points.append(pt)
					

			use_points = np.array(use_points)

			if use_points.size == 0: return None

			point_mass = np.rint(np.average(use_points, axis=0)).astype(np.int32)

			x_of_point_mass = point_mass[0, 0]

			if visualize:
				cv2.circle(img, tuple(point_mass[0].tolist()), 5, (255, 255, 255), 1)
				image_print(img, "Hard turn")


			# is the point mass on the left?
			if x_of_point_mass < img_shape[1] / 2:
				# yes! publish that we should do a hard left
				return True
			else:
				# no, publish that we should do a hard right
				return False


			# for idx, pt in enumerate(second_self_intersections): 
			# 	print(repr(tuple(pt.tolist())))	
			# 	cv2.circle(img, tuple(pt[0].tolist()), 5, (255,255,255), 1)
			# cv2.circle(img, tuple(point_mass[0].tolist()), 5, (255, 255, 255), 1)

			# don't bother trying to add more points to consideration on the nearly horizontal case
			# just add the lower classifications, the overall intersection, and the horizontal point mass
			points_in_trajectory = []
			y_boundary = average_point[0][1]
			for intercept in range(0, img_shape[0] - y_boundary, horizontal_line_step):
				yval = img_shape[0] - intercept

				new_filter = lower_intersections_to_use[:,1] == yval
				new_point = np.average(lower_intersections_to_use[new_filter,:], axis=0).astype(np.int32)
				points_in_trajectory.append(new_point)

			points_in_trajectory.append(tuple(average_point[0].tolist()))
			points_in_trajectory.append(tuple(point_mass[0].tolist()))


		else:
			# no, either the track is completely straight or it has some curve (but not near 90deg turn)

			# finds the points intersecting with the upper classification and the horizontal lines
			filter_upper = upper_intersections[:,:,1] < average_point[0][1]

			filter_upper = upper_intersections[:,:,1] > -10 # should always be nonneg
			upper_intersections_to_use = upper_intersections[filter_upper, :]
			# draw them (but not anymore)
			# for idx, pt in enumerate(upper_intersections_to_use): cv2.circle(img, tuple(pt), 5, (255, 0, 255), 1)

			# create a trajectory using the average intersection point moving upwards
			yboundary = average_point[0][1]
			# points_in_trajectory = [tuple(average_point[0].tolist())]
			points_in_trajectory = list()

			points_in_first_traj = list()
			points_in_avg_intersect = [tuple(average_point[0].tolist())]
			points_in_second_traj = list()

			num_second_class_is_below = 0

			for intercept in range(0, int(img_shape[0]*pixel_cutoff), horizontal_line_step):
				# y = 0 is at the top
				# lines = np.vstack((lines, np.array([[img_shape[0] - intercept, np.pi/2]])[np.newaxis, :, :]))
				yval = img_shape[0] - intercept

				lower_filter = lower_intersections_to_use[:,1] == yval
				upper_filter = upper_intersections_to_use[:,1] == yval

				to_avg_lower = lower_intersections_to_use[lower_filter,:]
				to_avg_upper = upper_intersections_to_use[upper_filter,:]

				if to_avg_lower.size == 0: continue

				new_point_lower = np.average(to_avg_lower, axis=0).astype(np.int32)
				new_point_upper = np.average(to_avg_upper, axis=0).astype(np.int32)

				if detected_single_lane:
					# we see a single lane so average all the points together
					if visualize: print("we detected a single line")
					new_point = np.average(np.array([new_point_lower, new_point_upper]), axis=0).astype(np.int32)

					# are the points in the orange mask?
					if (mask[new_point[1], new_point[0]] > 200):
						points_in_trajectory.append(tuple(new_point.tolist()))

				else: # the two classes of lines are separate lines
					# are the new points in the orange mask?
					if (mask[new_point_lower[1], new_point_lower[0]] > 200): 
						# yes, add it to trajectory
						points_in_first_traj.append(tuple(new_point_lower.tolist()))

					if (mask[new_point_upper[1], new_point_upper[0]] > 200): 
						# yes, add it to trajectory
						if yval > yboundary: # is the less steep class below the average intersection?
							num_second_class_is_below += 1
						points_in_second_traj.append(tuple(new_point_upper.tolist()))
				
			if detected_single_lane:
				# sort the points from nearest to the bottom to the farthest only when we're not doing a 90 degree turn
				points_in_trajectory = sorted(points_in_trajectory, key=lambda p: p[1], reverse=True) 	

			else:
				# are more than half the points in the second class below the boundary?
				second_class_is_below = (num_second_class_is_below > 0.5 * len(points_in_second_traj))

				# sort from nearest to bottom if the second class is not below. Otherwise sort top to bottom
				points_in_second_traj = sorted(points_in_second_traj, key=lambda p: p[1], reverse=(not second_class_is_below))
				points_in_trajectory = points_in_first_traj + points_in_avg_intersect + points_in_second_traj


		# end if-else

		if visualize:
			# paint the trajectory on the image using a dark-to-teal gradient
			for idx, pt in enumerate(points_in_trajectory): cv2.circle(img, tuple(pt), 5, (255*idx/len(points_in_trajectory), 255*idx/len(points_in_trajectory), 0), 1)
			
			image_print(img, "image with trajectory")
		# Now that we have a trajectory, we should return the points
		return points_in_trajectory


	# we couldn't find any lines :(
	return None

# if __name__ == '__main__':
# 	imgg = cv2.imread("./test_images_more_cones/IMG_7709.jpg")
# 	cd_color_segmentation(imgg)

# if __name__ == '__main__':
# 	imgg = cv2.imread("./test_images_more_cones/IMG_7709.jpg")
# 	cd_color_segmentation(imgg)


if __name__ == '__main__':
	_img = cv2.imread("./test_images_track/anger.png")

	# orange mask
	lower_bounds = (10, 10, 120)
	upper_bounds = (30,255, 255)

	
	# brown mask
	# lower_bounds = (10, 0, 0)
	# upper_bounds = (50,255, 110)

	# lf_color_segmentation(_img, template=[lower_bounds, upper_bounds], visualize=True)
	lf_color_segmentation(_img, visualize=True, horizontal_angle_margin=20*np.pi/180)
