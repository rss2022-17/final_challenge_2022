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
		lower_bounds = (5, 200, 100)
		upper_bounds = (35, 255, 255)
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

def intersection(line1, line2):
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


def segmented_intersections(lines):
	# https://stackoverflow.com/questions/46565975/find-intersection-point-of-two-lines-drawn-using-houghlines-opencv
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) 

    return intersections

def self_intersections(lines):

	intersections = []



def lf_color_segmentation(img, template=None, pct=0.25): #pct specifies which portion of the image to look in
	"""
	Implement the cone detection using color segmentation algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected. BGR.
		template_file_path; List of two tuples: (low values, high values) in where each value is (hue, sat, val)
		pct: float; specifies which portion of the image to look in for the line
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
	"""
	########## YOUR CODE STARTS HERE ##########
	image_print(img)
	x = y = w = h = 0
	# image_print(img) #prints image
	# bounding values: change for different lighting conditions in HSV
	if template is not None:
		lower_bounds = template[0]
		upper_bounds = template[1]
	else:
		lower_bounds = (5, 190, 100)
		upper_bounds = (40, 255, 255)
		
		# lower_bounds = (5, 200, 100)
		# upper_bounds = (35, 255, 255)
		
	hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #converts bgr to hsv
	img_shape = hsv_img.shape

	start_x = int(np.floor(img_shape[0]*(1-pct)))
	hsv_img_cropped = hsv_img[start_x:,:]
	mask = cv2.inRange(hsv_img_cropped, lower_bounds, upper_bounds) #creates binary image with 1 = within bounds given
	element = np.ones((5, 5), np.uint8) #kernel for erosion/dilation
	erosion_dst = cv2.erode(mask, element, iterations=1) #shrinks mask area down
	mask = cv2.dilate(erosion_dst, element, iterations=1) #increases mask area
	# image_print(mask, "mask")

	### Now that we have a mask, do line detection
	canny_img = cv2.Canny(mask, 100, 200)

	# HoughLines(img, rho, theta, threshold, lines,)
	lines = cv2.HoughLines(canny_img, 1, np.pi/180, 50, None, 0, 0)

	# Add horizontal lines to the image [r = y-int, theta = pi/2]
	for intercept in range(0, int(img_shape[0]/2), 60):
		# y = 0 is at the top
		lines = np.vstack((lines, np.array([[img_shape[0] - intercept, np.pi/2]])[np.newaxis, :, :]))


	if lines is not None:
		# Create three groups of lines
		segmented = segment_angle_kmeans(lines, k=3)

		# sort by theta, don't reverse so we consider high angles first
		segmented = sorted(segmented, key=lambda g: g[0][0][1], reverse=False) 

		# add the different groups by color to the base image
		group_colors = [(0, 255, 0), (255, 0, 0), (25, 25, 25)]
		for g_idx, group in enumerate(segmented):
			for l_idx, l in enumerate(group):
				rho = l[0][0]
				theta = l[0][1]
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a * rho
				y0 = b * rho
				pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
				pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
				
				cv2.line(img, pt1, pt2, group_colors[g_idx], 2, cv2.LINE_AA)


		image_print(img, "lines")

		# find where the lines in different groups intersect eachother
		intersections = segmented_intersections(segmented)
		for pt in intersections: cv2.circle(img, tuple(pt[0]), 5, (255, 0, 255), 1)


	else:
		print("There are no lines :(")

	image_print(img, "image with lines")


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
		# cv2.rectangle(img, (x, y), (x+w,y+h), (255, 0, 0), 2) #adds box onto image
	#image_print(img)
	#bounding_box = ((x,y+start_y),(x+w,y+h+start_y))
	bounding_box = ((x,y+start_x), (x+w, y+h+start_x))
	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box

# if __name__ == '__main__':
# 	imgg = cv2.imread("./test_images_more_cones/IMG_7709.jpg")
# 	cd_color_segmentation(imgg)

# if __name__ == '__main__':
# 	imgg = cv2.imread("./test_images_more_cones/IMG_7709.jpg")
# 	cd_color_segmentation(imgg)


if __name__ == '__main__':
	_img = cv2.imread("./test_images_track/city-driving-line-following.png")
	lf_color_segmentation(_img, pct=1)