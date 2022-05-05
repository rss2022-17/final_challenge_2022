import cv2
import numpy as np
import pdb

def lf_color_segmentation(img, template=None, pct=0.25): #pct specifies which portion of the image to look in
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
	img_shape = hsv_img.shape
	start_y = np.floor(img_shape[1]*(1-pct))
	hsv_img_cropped = hsv_img[:,start_y:]
	mask = cv2.inRange(hsv_img_cropped, lower_bounds, upper_bounds) #creates binary image with 1 = within bounds given
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
		M = cv2.moments(cnt)
		cX = int(M["m10"]/ M["m00"])
		cY = int(M["m01"]/ M["m00"])
		return (cX, cY)

	#image_print(img)
	return None
	bounding_box = ((x,y+start_y),(x+w,y+h+start_y))

	########### YOUR CODE ENDS HERE ###########

	# Return bounding box
	return bounding_box