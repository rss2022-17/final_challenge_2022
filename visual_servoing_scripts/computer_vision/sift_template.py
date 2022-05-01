import cv2
import imutils
import numpy as np
import pdb


# DELETE THESE IMPORTS
import csv
import ast
from matplotlib import pyplot as plt

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

def image_print(img):
	"""
	Helper function to print out images, for debugging.
	Press any key to continue.
	"""
	winname = "Image"
	cv2.namedWindow(winname)        # Create a named window
	cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
	cv2.imshow(winname, img)
	cv2.waitKey()
	cv2.destroyAllWindows()

def cd_sift_ransac(img, template):
	"""
	Implement the cone detection using SIFT + RANSAC algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	# Minimum number of matching features
	MIN_MATCH = 10
	# Create SIFT
	sift = cv2.xfeatures2d.SIFT_create()

	# Compute SIFT on template and test image
	kp1, des1 = sift.detectAndCompute(template,None)
	kp2, des2 = sift.detectAndCompute(img,None)

	# Find matches
	bf = cv2.BFMatcher()
	matches = bf.knnMatch(des1,des2,k=2)

	# Find and store good matches
	good = []
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			good.append(m)

	# If enough good matches, find bounding box
	if len(good) > MIN_MATCH:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		# Create mask
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
		matchesMask = mask.ravel().tolist()

		h, w = template.shape
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

		########## YOUR CODE STARTS HERE ##########

		x_min = y_min = x_max = y_max = 0

		########### YOUR CODE ENDS HERE ###########

		# Return bounding box
		return ((x_min, y_min), (x_max, y_max))
	else:

		print ("[SIFT] not enough matches; matches: ", len(good))

		# Return bounding box of area 0 if no match found
		return ((0,0), (0,0))

def cd_template_matching(img, template, use_method=cv2.TM_CCOEFF_NORMED, image_name=None):
	"""
	Implement the cone detection using template matching algorithm
	Input:
		img: np.3darray; the input image with a cone to be detected
	Return:
		bbox: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
				(x1, y1) is the bottom left of the bbox and (x2, y2) is the top right of the bbox
	"""
	template_canny = cv2.Canny(template, 50, 200)

	# Perform Canny Edge detection on test image
	grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_canny = cv2.Canny(grey_img, 50, 200)

	# Get dimensions of template
	(img_height, img_width) = img_canny.shape[:2]

	# Keep track of best-fit match
	best_match = None

	# Loop over different scales of image
	for scale in np.linspace(1.5, .5, 50):
		# Resize the image
		resized_template = imutils.resize(template_canny, width = int(template_canny.shape[1] * scale))
		(h,w) = resized_template.shape[:2]
		# Check to see if test image is now smaller than template image
		if resized_template.shape[0] > img_height or resized_template.shape[1] > img_width:
			continue

		########## YOUR CODE STARTS HERE ##########
		# Use OpenCV template matching functions to find the best match
		# across template scales.

		########## USED OPENCV DOCUMENTATION AS REFERENCE ####
		## https://docs.opencv.org/3.4/d4/dc6/tutorial_py_template_matching.html
		##
		##
		## methods: 'cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR','cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED'
		##
		##  If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
		##

		look_at_min = use_method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

		result = cv2.matchTemplate(img_canny, resized_template, method=use_method)
		min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

		# used TM_CCOEFF so
		if look_at_min:
			top_left = min_loc # ((xmin, ymin))
		else:
			top_left = max_loc # ((xmin, ymin))

		bottom_right = (int(top_left[0] + w), int(top_left[1]+h))
		# bottom_left = (top_left[0], int(top_left[1] + h/scale))
		# top_right = (int(top_left[0] + w/scale), top_left[1])

		temp_bounding_box = (top_left, bottom_right)
		# print(temp_bounding_box)

		if best_match is not None:
			val, b_box = best_match

			if look_at_min:
				if min_val <= val:
					best_match = (min_val, temp_bounding_box)
				else:
					bounding_box = b_box
			else:
				if max_val >= val:
					best_match = (max_val, temp_bounding_box)

			if max_val >= val:
				best_match = (max_val, temp_bounding_box)
			else:
				bounding_box = b_box

		else:
			if look_at_min:
				best_match = (min_val, temp_bounding_box)
			else:
				best_match = (max_val, temp_bounding_box)
			
			bounding_box = temp_bounding_box

		# Remember to resize the bounding box using the highest scoring scale
		# x1,y1 pixel will be accurate, but x2,y2 needs to be correctly scaled
		# bounding_box = ((0,0),(0,0))
		########### YOUR CODE ENDS HERE ###########

	if __name__ == "__main__":
		cv2.rectangle(img, bounding_box[0], bounding_box[1], 255, 2)
		# image_print(img)

		if image_name is not None:
			cv2.imwrite(image_name, img)
			cv2.waitKey(0)

		pass
	return bounding_box



### DELETE AFTER THIS
# File paths
cone_csv_path = "./test_images_cone/test_images_cone.csv"
citgo_csv_path = "./test_images_citgo/test_citgo.csv"
localization_csv_path="./test_images_localization/test_localization.csv"

cone_template_path = './test_images_cone/cone_template.png'
citgo_template_path = './test_images_citgo/citgo_template.png'
localization_template_path='./test_images_localization/basement_fixed.png'

cone_score_path = './scores/test_scores_cone.csv'
citgo_score_path = './scores/test_scores_citgo.csv'
localization_score_path = './scores/test_scores_map.csv'

def iou_score(bbox1, bbox2):
    """
    Return the IoU score for the two bounding boxes
    Input:
        bbox1: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
                (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
        bbox2: ((x1, y1), (x2, y2)); the bounding box of the cone, unit in px
                (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
    Return:
        score: float; the IoU score
    """
    # First check bbox is coming in the correct order
    if bbox1[0][0] > bbox1[1][0] or bbox1[0][1] > bbox1[1][1]:
        print ("Check that you are returning bboxes as ((xmin, ymin),(xmax,ymax))")
    # Determine intersection rectangle
    x_int_1 = max(bbox1[0][0], bbox2[0][0])
    y_int_1 = max(bbox1[0][1], bbox2[0][1])
    x_int_2 = min(bbox1[1][0], bbox2[1][0])
    y_int_2 = min(bbox1[1][1], bbox2[1][1])

    # Compute area of intersection
    
    # Check if the bounding boxes are disjoint (no intersection)
    if x_int_2 - x_int_1 < 0 or y_int_2 - y_int_1 < 0:
        area_int = 0
    else:
        area_int = (x_int_2 - x_int_1 + 1) * (y_int_2 - y_int_1 + 1)
    
    # Compute area of both bounding boxes
    area_bbox1 = (bbox1[1][0] - bbox1[0][0] + 1) * (bbox1[1][1] - bbox1[0][1] + 1)
    area_bbox2 = (bbox2[1][0] - bbox2[0][0] + 1) * (bbox2[1][1] - bbox2[0][1] + 1)

    # Compute area of union
    area_union = float(area_bbox1 + area_bbox2 - area_int)

    # Compute and return IoU score
    score = area_int / area_union

    # Reject negative scores
    if score < 0:
        score = 0

    return score

def test_algorithm(detection_func, csv_file_path, template_file_path, swap=False, method=cv2.TM_CCOEFF):
    """
    Test a cone detection function and return the average score based on all the test images
    Input:
        detection_func: func; the cone detection function that takes the np.3darray
                as input and return (bottom, left, top, right) as output
        csv_file_path: string; the path to the csv file
        template_file_path: string, path to template file
        swap: Optional tag for indicating the template_file is really the background file
        For the map template matching, these need to be inverted
    Return:
        scores: dict; the score for each test image
    """
    # Keep track of scores
    scores = {}
    # Open test images csv
    with open(csv_file_path) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        # Iterate through all test images
        for row in csvReader:
            # Find image path and ground truth bbox
            img_path = row[0]
            bbox_true = ast.literal_eval(row[1])
            if not swap:
                img = cv2.imread(img_path)
                template = cv2.imread(template_file_path, 0)
            else:
                template = cv2.imread(img_path, 0)
                img = cv2.imread(template_file_path)
            # Detection bbox
            bbox_est = detection_func(img, template, use_method=method)
            score = iou_score(bbox_est, bbox_true)
            
            # Add score to dict
            scores[img_path] = score

    # Return scores
    return scores


if __name__ == "__main__":

	## Testing Template Matching
	# img = cv2.imread("./test_images_cone/test5.jpg")
	# template = cv2.imread("./test_images_cone/cone_template.png", 0)

	# img = cv2.imread("./test_images_citgo/citgo14.jpeg")
	# template = cv2.imread("./test_images_citgo/citgo_template.png", 0)

	img = cv2.imread("./test_images_localization/basement_fixed.png")
	template = cv2.imread("./test_images_localization/map_scrap6.png", 0)


	bounding_box = cd_template_matching(img, template, image_name="./result_images/local_6.png")

	methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
            'cv2.TM_CCORR_NORMED',
			'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

	test_types = ["cone", "localization", "citgo"]
	number_of_tests = {
		"cone": 20,
		"localization": 9,
		"citgo": 14
	}
	paths = [cone_csv_path, localization_csv_path, citgo_csv_path]
	templates = [cone_template_path, localization_template_path, citgo_template_path]
	swaps = [False, True, False]

	all_scores = {}
	
	if False:
		# don't run full test

		for test_idx in range(3):
			print("Running Test: "+str(test_idx))
			for m in methods:
				print("Using Method: "+m)
				meth = eval(m)
				scores = test_algorithm(cd_template_matching, paths[test_idx], templates[test_idx], swap=swaps[test_idx], method=meth)


				t_type = test_types[test_idx]

				# all_scores[str(test_idx)+"_"+m] = scores

				if scores:
					for (img, val) in scores.iteritems():
						# print((img, val))
						if t_type == "cone":
							test_number = img[len('./test_images_cone/test'):-len('.jpg')]
						elif t_type == "localization":
							test_number = img[len('./test_images_localization/map_scrap'):-len('.png')]
						else:
							test_number = img[len('./test_images_citgo/citgo'):-len('.jpeg')]

						# print("\tTest number is: "+test_number)

						all_scores[t_type+"_"+test_number+"_"+m] = val


	
	# with open('my_test_file.txt', 'w') as f:
	# 	for (t, s) in all_scores.iteritems():
	# 			to_write = "=== "+t+" ===\n"
	# 			f.write(to_write)
	# 			for (img, val) in s.iteritems():
	# 				f.write(str((img, val)) + "\n")
		with open('template_matching_results.csv', 'w') as csvfile:
			writer = csv.writer(csvfile)

			writer.writerow(['Test'] + methods)

			for t in test_types:
				for idx in range(number_of_tests[t]):

					row_to_write = [t + "_" + str(idx+1)]
					for m in methods:
						row_to_write += [all_scores[t+"_"+str(idx+1)+"_"+m]]
					
					writer.writerow(row_to_write)


