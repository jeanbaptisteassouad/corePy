import cv2
import numpy as np

cropRectPoints = []
cropping = False
image = 0

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global cropRectPoints, cropping, image

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		cropRectPoints = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		cropRectPoints.append((x, y))
		cropping = False

		# draw a rectangle around the region of interest
		imageTmp = image.copy()
		cv2.rectangle(imageTmp, cropRectPoints[0], cropRectPoints[1], (0, 0, 0), 2)
		cv2.imshow("image", imageTmp)

	elif cropping == True:
		imageTmp = image.copy()
		cv2.rectangle(imageTmp, cropRectPoints[0], (x, y), (0, 0, 0), 2)
		cv2.imshow("image", imageTmp)

def apply_crop():
	height,width,channel = image.shape
	minY = min(cropRectPoints[0][1],cropRectPoints[1][1])
	minY = max(0,minY)
	maxY = max(cropRectPoints[0][1],cropRectPoints[1][1])
	maxY = min(height,maxY)
	minX = min(cropRectPoints[0][0],cropRectPoints[1][0])
	minX = max(0,minX)
	maxX = max(cropRectPoints[0][0],cropRectPoints[1][0])
	maxX = min(width,maxX)
	table = image[minY:maxY, minX:maxX]
	cv2.imshow('image', table)
	cv2.waitKey(0)
	return table


def main():
	global image
	global cropRectPoints
	image = cv2.imread('test.png')
	cv2.namedWindow('image')
	cv2.setMouseCallback('image', click_and_crop)
	cv2.imshow('image',image)
	cropRectPoints = []
	while (len(cropRectPoints) != 2):
		keyPressed = cv2.waitKey(0)
		if keyPressed == 1048603:
			exit()
	apply_crop()
	cv2.destroyAllWindows()
	pass

if __name__ == '__main__':
	main()