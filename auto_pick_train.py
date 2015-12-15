import sys
import os
import scipy.spatial.distance as scipyDistance
import cv2
import numpy as np
from Feature import Feature
import random

def image_to_list(image):
	# f = Feature()
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# return f.projection_histogram(image,gray)

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

	return moy

def get_training_image(path):
	listImageAlreadyUse = []
	lastVariance = -1

	listNameFile = []
	for filename in os.listdir(path):
		listNameFile.append( filename )

	random.shuffle(listNameFile)

	for x in range(0,len(listNameFile)):
		print lastVariance
		if x==0:
			listImageAlreadyUse.append( [image_to_list( cv2.imread(path+listNameFile[x]) ) , x] )
		else:
			imageTmp = image_to_list( cv2.imread(path+listNameFile[x]) )
			newVariance = computeVariance(listImageAlreadyUse,imageTmp)

			if newVariance > lastVariance:
				lastVariance = newVariance
				listImageAlreadyUse.append( [imageTmp , x] )
				pass
	print lastVariance
	# print len(listImageAlreadyUse)
	ans = []
	for x in range(0,len(listImageAlreadyUse)):
		# print listNameFile[listImageAlreadyUse[x][1]]
		ans.append(listNameFile[listImageAlreadyUse[x][1]])
		pass

	return ans


def get_median_len(listNameFile,path):
	listDistance = []
	for x in range(0,len(listNameFile)):
		print x
		listImage = image_to_list( cv2.imread(path+listNameFile[x]) )
		for y in range(x,len(listNameFile)):
			if y!=x:
				listImageTmp = image_to_list( cv2.imread(path+listNameFile[y]) )
				listDistance.append( scipyDistance.euclidean(listImageTmp , listImage) )
				pass
			pass
		pass

	listDistance.sort()
	return listDistance[len(listDistance)/2 - 1]
	# return sum(listDistance)/float(len(listDistance))

def build_epsilon_network(epsilon,listNameFile,path):
	listEpsilonNetwork = []
	for x in range(0,len(listNameFile)):
		listImage = image_to_list( cv2.imread(path+listNameFile[x]) )
		shouldAdd = True
		for y in range(0,len(listEpsilonNetwork)):
			tmp = scipyDistance.euclidean(listEpsilonNetwork[y][0] , listImage)
			if tmp < epsilon:
				shouldAdd = False
				break
		if shouldAdd:
			listEpsilonNetwork.append([listImage,x])
			pass

	return listEpsilonNetwork

def check_epsilon_network(listEpsilonNetwork,epsilon,listNameFile,path):
	print "Check intern distance"
	for x in range(0,len(listEpsilonNetwork)):
		for y in range(x,len(listEpsilonNetwork)):
			if x!=y:
				if scipyDistance.euclidean(listEpsilonNetwork[y][0],listEpsilonNetwork[x][0]) < epsilon:
					return False
	print "Check succeed"
	pass

def number_under_epsilon(listEpsilonNetwork,epsilon,listNameFile,path):
	for x in range(0,len(listEpsilonNetwork)):
		cpt = 0
		for y in range(0,len(listNameFile)):
			listImage = image_to_list( cv2.imread(path+listNameFile[y]) )
			tmp = scipyDistance.euclidean(listEpsilonNetwork[x][0] , listImage)
			if tmp < epsilon and tmp != 0:
				cpt += 1
		listEpsilonNetwork[x].append(cpt)

	return listEpsilonNetwork


def test(path):
	listNameFile = []
	for filename in os.listdir(path):
		listNameFile.append( filename )

	random.shuffle(listNameFile)

	# print get_median_len(listNameFile,path)
	listEpsilonNetwork = build_epsilon_network( 1484 ,listNameFile,path)

	# check_epsilon_network(listEpsilonNetwork, 1484 ,listNameFile,path)

	listEpsilonNetwork = number_under_epsilon(listEpsilonNetwork, 1484 ,listNameFile,path)

	for x in range(0,len(listEpsilonNetwork)):
		print listNameFile[ listEpsilonNetwork[x][1] ], listEpsilonNetwork[x][2]
		pass

	pass

def main():
	# print get_training_image("png/10/HPC-T4-2013-GearsAndSprockets-GB/")
	test("png/10/HPC-T4-2013-GearsAndSprockets-GB/")


if __name__ == '__main__':
	main()