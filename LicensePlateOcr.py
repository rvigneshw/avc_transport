import cv2
import image
import datetime
from pytesseract import image_to_string
import numpy as np

class LicensePlateOcr:
	def __init__(self):
		this.self=self

	def read_image(imagefile):
		img = imagefile
		# Creating a Named window to display image
		# Display image

		# RGB to Gray scale conversion
		img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
		# Creating a Named window to display image
		# Display Image

		# Noise removal with iterative bilateral filter(removes noise while preserving edges)
		noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
		# Creating a Named window to display image
		# Display Image

		# Histogram equalisation for better results
		equal_histogram = cv2.equalizeHist(noise_removal)
		# Creating a Named window to display image
		# Display Image

		# Morphological opening with a rectangular structure element
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
		morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)
		# Creating a Named window to display image
		# Display Image

		# Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
		sub_morp_image = cv2.subtract(equal_histogram,morph_image)
		# Creating a Named window to display image
		# Display Image

		# Thresholding the image
		ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
		# Creating a Named window to display image
		# Display Image

		# Applying Canny Edge detection
		canny_image = cv2.Canny(thresh_image,250,255)
		# Creating a Named window to display image
		# Display Image
		canny_image = cv2.convertScaleAbs(canny_image)

		# dilation to strengthen the edges
		kernel = np.ones((3,3), np.uint8)
		# Creating the kernel for dilation
		dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
		# Creating a Named window to display image
		# Displaying Image

		# Finding Contours in the image based on edges
		new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
		# Sort the contours based on area ,so that the number plate will be in top 10 contours
		screenCnt = None
		# loop over our contours
		for c in contours:
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			if len(approx) == 4:  # Select the contour with 4 corners
				NumberPlateCnt = approx #This is our approx Number Plate Contour
				break

		#
		# for c in contours:
		# 	# approximate the contour
		# 	peri = cv2.arcLength(c, True)
		# 	approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
		# 	# if our approximated contour has four points, then
		# 	# we can assume that we have found our screen
		# 	if len(approx) == 4:  # Select the contour with 4 corners
		# 		screenCnt = approx
		# 		break


		final = cv2.drawContours(img, [NumberPlateCnt], -1, (0, 255, 0), 3)

		# Masking the part other than the number plate
		mask = np.zeros(img_gray.shape,np.uint8)
		new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1,)
		new_image = cv2.bitwise_and(img,img,mask=mask)
		# cv2.namedWindow("Final_image",cv2.WINDOW_NORMAL)
		# cv2.imshow("Final_image",new_image)
		number=image_to_string(new_image,config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
		return number
		# status=DB.check_number(number)
		# st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
		# DB.register_log(number,st,status)
		# cv2.waitKey(0) 
