import sys
import os
import scipy.spatial.distance as scipyDistance
import cv2
import numpy as np

def image_to_list(image):
	ans = np.zeros( 3*len(image)*len(image[0]) )
	cpt = 0
	for y in range(0,len(image)):
		for x in range(0,len(image[0])):
			ans[cpt] = image[y][x][0]
			cpt += 1
			ans[cpt] = image[y][x][1]
			cpt += 1
			ans[cpt] = image[y][x][2]
			cpt += 1
			pass
		pass
	return ans


def computeVariance(listImageAlreadyUse,newOne):
	ans = []

	for x in range(0,len(listImageAlreadyUse)):
		ans.append( scipyDistance.euclidean(listImageAlreadyUse[x][0] , newOne) )
		pass
	for x in range(0,len(listImageAlreadyUse)):
		for y in range(x+1,len(listImageAlreadyUse)):
			ans.append( scipyDistance.euclidean(listImageAlreadyUse[x][0] , listImageAlreadyUse[y][0]) )
		pass
	if ans==[]:
		return 0

	variance = 0
	moy = sum(ans)/len(ans)
	for x in range(0,len(ans)):
		variance += pow(ans[x] - moy  ,2)
		pass
	variance /= len(ans)
	variance = pow(variance,0.5)

	return moy*variance

def get_training_image(path):
	listImageAlreadyUse = []
	lastVariance = -1

	listNameFile = []
	for filename in os.listdir(path):
		listNameFile.append( filename )

	for x in range(0,len(listNameFile)):
		if x==0:
			listImageAlreadyUse.append( [image_to_list( cv2.imread(path+listNameFile[x]) ) , x] )
		else:
			imageTmp = image_to_list( cv2.imread(path+listNameFile[x]) )
			newVariance = computeVariance(listImageAlreadyUse,imageTmp)

			if newVariance > lastVariance:
				lastVariance = newVariance
				listImageAlreadyUse.append( [imageTmp , x] )
				pass

	# print len(listImageAlreadyUse)
	ans = []
	for x in range(0,len(listImageAlreadyUse)):
		# print listNameFile[listImageAlreadyUse[x][1]]
		ans.append(listNameFile[listImageAlreadyUse[x][1]])
		pass

	return ans


def main():
	print get_training_image("png/10/HPC-T4-2013-GearsAndSprockets-GB/")


if __name__ == '__main__':
	main()