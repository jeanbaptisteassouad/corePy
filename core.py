import cv2
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import projection as proj
import scipy.spatial.distance as scipyDistance


cropRectPoints = []
cropping = False
image = 0
gray = 0

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
        cv2.rectangle(imageTmp, cropRectPoints[0], cropRectPoints[1], (0, 0, 0), 1)
        cv2.imshow("image", imageTmp)

    elif cropping == True:
        imageTmp = image.copy()
        cv2.rectangle(imageTmp, cropRectPoints[0], (x, y), (0, 0, 0), 1)
        cv2.imshow("image", imageTmp)

def dummyCallBack(event, x, y, flags, param):
    pass

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
    return table

# Cote :        0
#            IIIIIIII
#       1 -> I page I <- 3
#            IIIIIIII
#               2
# Start allWhiteFrame, travel all allWhiteFrame until we get to the last allWhiteFrame before a non allWhiteFrame
# Return allWhiteFrame
def getNextPreFrame(frame,cote):
    frameArray = [ [frame[0][0],frame[0][1]] , [frame[1][0],frame[1][1]] ]
    if cote == 0:
        point = 0
        isY = 1
        translation = 1
    if cote == 1:
        point = 0
        isY = 0
        translation = 1
    if cote == 2:
        point = 1
        isY = 1
        translation = -1
    if cote == 3:
        point = 1
        isY = 0
        translation = -1

    while frameIsAllWhite(frameArray):
        frameArray[point][isY] += translation
        if frameIsDegenerate(frameArray) :
            return [(-1,-1),(-1,-1)]
        pass
    frameArray[point][isY] -= translation
    frame = [ (frameArray[0][0],frameArray[0][1]) , (frameArray[1][0],frameArray[1][1]) ]
    return frame

# Transform the big frame (0,0) (width-1,height-1) into an allWhiteFrame
def prepareImageGray():
    global image
    global gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height,width,channel = image.shape
    for x in range(0,width):
        gray[0][x] = 255
        gray[height-1][x] = 255
        pass
    for y in range(0,height):
        gray[y][0] = 255
        gray[y][width-1] = 255
        pass

#Start allWhiteFrame, travel all non allWhiteFrame until we get an other allWhiteFrame
def getNextFrame(frame,cote):
    frameArray = [ [frame[0][0],frame[0][1]] , [frame[1][0],frame[1][1]] ]
    if cote == 0:
        point = 0
        isY = 1
        translation = 1
    if cote == 1:
        point = 0
        isY = 0
        translation = 1
    if cote == 2:
        point = 1
        isY = 1
        translation = -1
    if cote == 3:
        point = 1
        isY = 0
        translation = -1

    frameArray[point][isY] += translation
    while not frameIsAllWhite(frameArray):
        frameArray[point][isY] += translation
        if frameIsDegenerate(frameArray) :
            return [(-1,-1),(-1,-1)]
            pass
        pass
    frame = [ (frameArray[0][0],frameArray[0][1]) , (frameArray[1][0],frameArray[1][1]) ]
    frame = getNextPreFrame(frame,cote)
    return frame

def frameIsDegenerate(frame):
    if frame[0][0] >= frame[1][0]:
        return True
    if frame[0][1] >= frame[1][1]:
        return True
    return False


def frameIsAllWhite(frame):
    global gray
    threshold = 255
    for y in range(frame[0][1],frame[1][1]+1):
        if gray[y][frame[0][0]] < threshold or gray[y][frame[1][0]] < threshold:
            return False
    for x in range(frame[0][0],frame[1][0]+1):
        if gray[frame[0][1]][x] < threshold or gray[frame[1][1]][x] < threshold:
            return False
    return True

def dichotomieRecursive(frame,deep):
    if frame[0][0] == -1:
        return []
    elif deep <= 0:
        return [frame]
    else:
        return [frame] + dichotomieRecursive(getNextFrame(frame,0),deep-1) + dichotomieRecursive(getNextFrame(frame,1),deep-1) + dichotomieRecursive(getNextFrame(frame,2),deep-1) + dichotomieRecursive(getNextFrame(frame,3),deep-1)

def dichotomie(deep):
    height,width,channel = image.shape
    prepareImageGray()
    frame = [(0,0) , (width-1,height-1)]
    frame = getNextPreFrame(frame,0)
    frame = getNextPreFrame(frame,1)
    frame = getNextPreFrame(frame,2)
    frame = getNextPreFrame(frame,3)

    dichoArray = dichotomieRecursive(frame,deep)
    return dichoArray

def dichotomieRecursiveMultiProcessor(frame,deep,output):
    if frame[0][0] == -1:
        output.put([])
    elif deep <= 0:
        output.put([frame])
    else:
        output.put([frame] + dichotomieRecursive(getNextFrame(frame,0),deep-1) + dichotomieRecursive(getNextFrame(frame,1),deep-1) + dichotomieRecursive(getNextFrame(frame,2),deep-1) + dichotomieRecursive(getNextFrame(frame,3),deep-1))


def dichotomieMultiProcessor(deep):
    height,width,channel = image.shape
    prepareImageGray()
    frame = [(0,0) , (width-1,height-1)]
    frame = getNextPreFrame(frame,0)
    frame = getNextPreFrame(frame,1)
    frame = getNextPreFrame(frame,2)
    frame = getNextPreFrame(frame,3)

    if deep <= 0:
        return [frame]

    # inputs and output queue for multiprocessing purposes
    output = mp.Queue()
    # instantiate 4 processes
    processes = [mp.Process( target=dichotomieRecursiveMultiProcessor, args=(getNextFrame(frame,x),deep-1,output) ) for x in range(0,4)]
    # run processes
    for process in processes:
        process.start()
    # get results
    res = []
    for process in processes:
        res += output.get()
        pass

    return [frame] + res

def dichotomieRecursiveMultiProcessorOnlyLeef(frame,deep,output):
    if frame[0][0] == -1:
        output.put([])
    elif deep <= 0:
        output.put([frame])
    else:
        output.put(dichotomieRecursive(getNextFrame(frame,0),deep-1) + dichotomieRecursive(getNextFrame(frame,1),deep-1) + dichotomieRecursive(getNextFrame(frame,2),deep-1) + dichotomieRecursive(getNextFrame(frame,3),deep-1))



def dichotomieMultiProcessorOnlyLeef(deep):
    height,width,channel = image.shape
    prepareImageGray()
    frame = [(0,0) , (width-1,height-1)]
    frame = getNextPreFrame(frame,0)
    frame = getNextPreFrame(frame,1)
    frame = getNextPreFrame(frame,2)
    frame = getNextPreFrame(frame,3)

    if deep <= 0:
        return [frame]

    # inputs and output queue for multiprocessing purposes
    output = mp.Queue()
    # instantiate 4 processes
    processes = [mp.Process( target=dichotomieRecursiveMultiProcessorOnlyLeef, args=(getNextFrame(frame,x),deep-1,output) ) for x in range(0,4)]
    # run processes
    for process in processes:
        process.start()
    # get results
    res = []
    for process in processes:
        res += output.get()
        pass

    return res


def displayDichoArray(dichoArray):
    for x in range(0,len(dichoArray)):
        cv2.rectangle(image, dichoArray[x][0], dichoArray[x][1], (255, 0, 0), 1)
        pass
    pass


# def isFrame1InsideOrCrossFrame2(frame1,frame2):
#     if frame1[1][0] < frame2[0][0] or frame1[1][1] < frame2[0][1]:
#         return False
#     if frame1[0][0] > frame2[1][0] or frame1[0][1] > frame2[1][1]:
#         return False
#     return True


# def removeRedundancy():
#     newDichoArray = []
#     for i in range(0,len(dichoArray)):
#         pass
#         for j in range(0,len(dichoArray)):
#             if j != j:
#                 if isFrame1InsideOrCrossFrame2(j,i):
#                     pass
#                 pass
#             pass
#     pass


# def handClassify(dichoArray):
#     for x in range(0,len(dichoArray)):
#         table = image[dichoArray[x][0][1]:dichoArray[x][1][1], dichoArray[x][0][0]:dichoArray[x][1][0]]
#         cv2.imshow('image', table)
#         cv2.waitKey(0)
#         pass
#     pass

def analyseImageByDichotomieInterative(deep, goodProj, frame):
    score = [-1,-1,-1,-1]
    frameToEval = [-1,-1,-1,-1]
    tableToEval = [-1,-1,-1,-1]
    for x in range(0,4):
        frameToEval[x] = getNextFrame(frame,x)
        if frameToEval[x][0][0] != -1:
            tableToEval[x] = image[frameToEval[x][0][1]:frameToEval[x][1][1], frameToEval[x][0][0]:frameToEval[x][1][0]]
            score[x] = scipyDistance.euclidean( proj.projectionHist(tableToEval[x]), goodProj )
            pass
        pass
    best = score[0]
    bestInd = 0
    for x in range(1,4):
        if score[x] != -1 and score[x] < best:
            best = score[x]
            bestInd = x
            pass
        pass
    if best == -1:
        return frame, best
        pass
    # cv2.imshow('image', tableToEval[bestInd])
    # cv2.waitKey(0)

    return frameToEval[bestInd], best



def analyseImageByDichotomie(deep, goodProj):
    height,width,channel = image.shape
    prepareImageGray()
    frame = [(0,0) , (width-1,height-1)]
    frame = getNextPreFrame(frame,0)
    frame = getNextPreFrame(frame,1)
    frame = getNextPreFrame(frame,2)
    frame = getNextPreFrame(frame,3)
    score = 100000

    while True:
        oldFrame = frame
        oldScore = score
        frame, score = analyseImageByDichotomieInterative(deep-1, goodProj, frame)
        if oldFrame == frame:
            break
        if score > oldScore:
            frame = oldFrame
            break

    table = image[frame[0][1]:frame[1][1], frame[0][0]:frame[1][0]]
    cv2.imshow('image', table)
    cv2.waitKey(0)

    return frame


def main():
    global image
    global cropRectPoints
    global gray
    image = cv2.imread('core/test.png')
    cv2.namedWindow('image')

    cv2.setMouseCallback('image', click_and_crop)
    cv2.imshow('image',image)
    cropRectPoints = []
    while (len(cropRectPoints) != 2):
      keyPressed = cv2.waitKey(0)
      if keyPressed == 1048603:
          exit()
    cv2.setMouseCallback('image',dummyCallBack)
    table = apply_crop()
    # cv2.imshow('image', table)
    # cv2.waitKey(0)
    goodProj = proj.projectionHist(table)

    # image = cv2.imread('core/test+.png')
    analyseImageByDichotomie(0,goodProj)

    cv2.destroyAllWindows()
    pass


    # displayDichoArray( dichotomieMultiProcessor(3) )
    # cv2.imshow('image',image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()