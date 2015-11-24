import numpy as np
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt

class Content(object):
    """docstring for Content"""
    def __init__(self):
        super(Content, self).__init__()


    def __new_leuven_dichotomie_get_all_frame(self, gray, deep, frame):
        sumLine = np.zeros(frame[1][1]+1 - frame[0][1])
        sumCol = np.zeros(frame[1][0]+1 - frame[0][0])
        for y in range(frame[0][1],frame[1][1]+1):
            for x in range(frame[0][0],frame[1][0]+1):
                sumLine[y-frame[0][1]] += gray[y][x]
                sumCol[x-frame[0][0]] += gray[y][x]
                pass
            pass

        noMoreOne = True
        pointLine = []
        for i in range(0,len(sumLine)):
            if sumLine[i] < (frame[1][0]+1 - frame[0][0])*255:
                if noMoreOne:
                    noMoreOne = False
                    pointLine.append( i-1 + frame[0][1])
                    pass
                pass
            elif not noMoreOne:
                noMoreOne = True
                pointLine.append( i + frame[0][1])

        noMoreOne = True
        pointCol = []
        for i in range(0,len(sumCol)):
            if sumCol[i] < (frame[1][1]+1 - frame[0][1])*255:
                if noMoreOne:
                    noMoreOne = False
                    pointCol.append( i-1 + frame[0][0])
                    pass
                pass
            elif not noMoreOne:
                noMoreOne = True
                pointCol.append( i + frame[0][0])

        dichoArray = []
        for y in range(0,len(pointLine)/2):
            for x in range(0,len(pointCol)/2):
                dichoArray.append( [(pointCol[2*x],pointLine[2*y]),(pointCol[2*x+1],pointLine[2*y+1])] )

        return dichoArray

    def __new_leuven_dichotomie_recursive(self, gray, deep, frame):
        if deep <= 0:
            return [frame]

        dichoArray = self.__new_leuven_dichotomie_get_all_frame(gray, deep, frame)

        if len(dichoArray) == 1 and dichoArray[0] == frame:
            return [frame]

        ans = []
        for x in range(0,len(dichoArray)):
            ans += self.__new_leuven_dichotomie_recursive(gray, deep-1, dichoArray[x])
            pass

        return ans


    def __new_leuven_dichotomie_recursive_multiprocessing(self, gray, deep, frame, output):
        if deep <= 0:
            output.put([frame])
            return

        dichoArray = self.__new_leuven_dichotomie_get_all_frame(gray, deep, frame)

        if len(dichoArray) == 1 and dichoArray[0] == frame:
            output.put([frame])
            return

        ans = []
        for x in range(0,len(dichoArray)):
            ans += self.__new_leuven_dichotomie_recursive(gray, deep-1, dichoArray[x])
            pass

        output.put(ans)
        return

    # Extract Table from a pdf page
    def new_leuven_dichotomie(self,image, gray, deep=10):
        height,width,channel = image.shape
        frame = [(0,0) , (width-1,height-1)]

        # return self.__new_leuven_dichotomie_recursive(gray, deep, frame)

        if deep <= 0:
            return [frame]

        dichoArray = self.__new_leuven_dichotomie_get_all_frame(gray, deep, frame)

        if len(dichoArray) == 1 and dichoArray[0] == frame:
            return [frame]

        output = mp.Queue()
        processes = [mp.Process( target=self.__new_leuven_dichotomie_recursive_multiprocessing, args=(gray, deep-1, dichoArray[x], output) ) for x in range(0,len(dichoArray))]
        for process in processes:
            process.start()
        res = []
        for process in processes:
            res += output.get()

        return res


    def __ouverture(self,matrice,kernel, itera):
        matriceTmp = cv2.erode(matrice, kernel, iterations=itera)
        return cv2.dilate(matriceTmp, kernel, iterations=itera)

    def __fermeture(self,matrice,kernel, itera):
        matriceTmp = cv2.dilate(matrice, kernel, iterations=itera)
        return cv2.erode(matriceTmp, kernel, iterations=itera)



    def __paris_dichotomie_recursive(self,matrice,deep,frame):
        if deep <= 0:
            return [frame]
        sumLine = np.zeros(frame[1][1]+1 - frame[0][1])
        sumCol = np.zeros(frame[1][0]+1 - frame[0][0])
        for y in range(frame[0][1],frame[1][1]+1):
            for x in range(frame[0][0],frame[1][0]+1):
                sumLine[y-frame[0][1]] += matrice[y][x]
                sumCol[x-frame[0][0]] += matrice[y][x]
                pass
            pass

        noMoreOne = True
        pointLine = []
        for i in range(0,len(sumLine)):
            if sumLine[i] > 0:
                if noMoreOne:
                    noMoreOne = False
                    pointLine.append( i-1 + frame[0][1])
                    pass
                pass
            elif not noMoreOne:
                noMoreOne = True
                pointLine.append( i + frame[0][1])

        noMoreOne = True
        pointCol = []
        for i in range(0,len(sumCol)):
            if sumCol[i] > 0:
                if noMoreOne:
                    noMoreOne = False
                    pointCol.append( i-1 + frame[0][0])
                    pass
                pass
            elif not noMoreOne:
                noMoreOne = True
                pointCol.append( i + frame[0][0])

        if len(pointLine) == 0 or len(pointCol) == 0:
            return [frame]

        pointLine = compute_best_point(pointLine,frame[1][1]+1 - frame[0][1])
        pointCol = compute_best_point(pointCol,frame[1][0]+1 - frame[0][0])

        dichoArray = []
        for y in range(0,len(pointLine)/2):
            for x in range(0,len(pointCol)/2):
                dichoArray.append( [(pointCol[2*x],pointLine[2*y]),(pointCol[2*x+1],pointLine[2*y+1])] )
        pass

        ans = []
        for x in range(0,len(dichoArray)):
            ans += self.__paris_dichotomie_recursive(matrice, deep-1, dichoArray[x])

        return ans



    # Extract layout and cells from a table
    def paris_dichotomie(self,image, gray, deep=10):
        height,width,channel = image.shape
        frame = [(0,0) , (width-1,height-1)]

        ret, matrice = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        matrice = cv2.Laplacian(matrice, cv2.CV_8U)


        kernel = np.matrix('1; 1; 1')
        matriceVerti = self.__ouverture(matrice,kernel,10)
        kernel = np.matrix('1 1 1')
        matriceVerti = self.__fermeture(matriceVerti,kernel,1)

        kernel = np.matrix('1 1 1')
        matriceHori = self.__ouverture(matrice,kernel,10)
        kernel = np.matrix('1; 1; 1')
        matriceHori = self.__fermeture(matriceHori,kernel,1)

        for y in range(0,height):
            for x in range(0,width):
                if x == 0 or x == width-1 or y == 0 or y == height-1:
                    matrice[y][x] = 0
                else:
                    matrice[y][x] -= min(int(matriceVerti[y][x]) + int(matriceHori[y][x]),255)
                pass
            pass

        retval , matrice = cv2.threshold(matrice,125,255,cv2.THRESH_BINARY)

        cv2.imshow("Image1", matrice)

        # matrice = cv2.bitwise_not(matrice)

        # dichoArray = self.new_leuven_dichotomie(image, matrice, deep=0)

        # for x in range(0,len(dichoArray)):
        #     cv2.rectangle(matrice, dichoArray[x][0], dichoArray[x][1], (125, 0, 0), 1)

        # Pour les colonnes
        kernel = np.matrix('1; 1; 1')
        matrice = self.__fermeture(matrice,kernel,height)
        # kernel = np.matrix('1 1 1')

        # isFirst = True
        # cpt = 0
        # listEspace = []
        # for x in range(0,width):
        #     if matrice[0][x] == 0:
        #         isFirst = False
        #         cpt += 1
        #     if matrice[0][x] == 255 and isFirst == False:
        #         isFirst = True
        #         listEspace.append( cpt )
        #         cpt = 0
        #         pass
        #     pass
        # average = sum(listEspace)/len(listEspace)
        # kernel = np.matrix('1 1 1')
        # matrice = self.__fermeture(matrice,kernel,average/2)


        # # Pour les lignes
        # kernel = np.matrix('1 1 1')
        # matrice = self.__fermeture(matrice,kernel,width)
        # kernel = np.matrix('1; 1; 1')

        # isFirst = True
        # cpt = 0
        # listEspace = []
        # for y in range(0,height):
        #     if matrice[y][0] == 0:
        #         isFirst = False
        #         cpt += 1
        #     if matrice[y][0] == 255 and isFirst == False:
        #         isFirst = True
        #         listEspace.append( cpt )
        #         cpt = 0
        #         pass
        #     pass
        # average = sum(listEspace)/len(listEspace)
        # kernel = np.matrix('1; 1; 1')
        # matrice = self.__fermeture(matrice,kernel,average/2)


        # dichoArray = self.__paris_dichotomie_recursive(matrice,1,frame)


        # for x in range(0,len(dichoArray)):
        #     cv2.rectangle(matrice, dichoArray[x][0], dichoArray[x][1], (125, 0, 0), 1)

        cv2.imshow("Image", matrice)
        cv2.waitKey(0)
        pass


def apply_threshold_to_point(point,dimension,threshold):
    if point == []:
        return point
    elif len(point) < 4:
        return point
    else:
        if (point[1] - point[0])/float(dimension) <= threshold or (point[2] - point[1])/float(dimension) <= threshold or (point[3] - point[2])/float(dimension) <= threshold:
            return apply_threshold_to_point([point[0]] + point[3:len(point)],dimension,threshold)
        else:
            return [ point[0], point[1]] + apply_threshold_to_point(point[2:len(point)],dimension,threshold)
    pass


def compute_next_point_ecart_type(point,dimension):
    list_interval = []
    ratio = -1
    for i in range(0,len(point)-1):
        list_interval.append( (point[i+1] - point[i])/float(dimension) )
        pass

    threshold = 0
    average_list_inter = sum(list_interval)/len(list_interval)
    variance_list_inter = 0
    for i in range(0,len(list_interval)):
        variance_list_inter += pow( list_interval[i] - average_list_inter, 2 )
        pass
    variance_list_inter /= len(list_interval)
    ecart_type_list_inter = pow(variance_list_inter,0.5)
    list_interval.sort()

    threshold = list_interval[0]

    point = apply_threshold_to_point(point,dimension,threshold)

    if ecart_type_list_inter != 0:
        ratio = average_list_inter / ecart_type_list_inter

    return point , ratio


def compute_best_point(point,dimension):
    ratio = 0
    maxRatio = 0
    best_point = []
    # while ratio != -1:
    #     point, ratio = compute_next_point_ecart_type(point,dimension)
    #     if ratio > maxRatio:
    #         maxRatio = ratio
    #         best_point = point


    # list_interval = []
    # for i in range(0,len(point)/2):
    #     list_interval.append( (point[2*i+1] - point[2*i])/float(dimension) )
    #     pass

    # average = sum(list_interval)/len(list_interval)

    # for i in range(0,len(point)/2):
    #     if (point[2*i+1] - point[2*i])/float(dimension) > average:
    #         best_point.append( point[2*i] )
    #         best_point.append( point[2*i+1] )
    #         pass
    #     pass
    return best_point



def main():
    C = Content()
    pass

if __name__ == '__main__':
    main()
