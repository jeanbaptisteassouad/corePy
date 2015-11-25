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


    def __set_border_to_black(self,matriceLigne,matriceCol,frame):
        for y in range(0 , len(matriceCol) ):
            matriceCol[y][0] = 0
            matriceCol[y][frame[1][0] - frame[0][0]] = 0


        for x in range(0 ,len(matriceLigne[0]) ):
            matriceLigne[0][x] = 0
            matriceLigne[frame[1][1] - frame[0][1]][x] = 0

        return matriceLigne, matriceCol



    def __paris_dichotomie_recursive(self,matriceOrigine,deep,frame,thresholdWidth):
        if deep <= 0:
            return [frame]

        matrice = matriceOrigine[ frame[0][1]:frame[1][1]+1 , frame[0][0]:frame[1][0]+1 ]

        # Pour les colonnes
        kernel = np.matrix('1; 1; 1')
        matriceCol = self.__fermeture(matrice,kernel,frame[1][1]+1 - frame[0][1])
        kernel = np.matrix('1 1 1')
        matriceCol = self.__fermeture(matriceCol,kernel,thresholdWidth)


        # Pour les lignes
        kernel = np.matrix('1 1 1')
        matriceLigne = self.__fermeture(matrice,kernel,frame[1][0]+1 - frame[0][0])
        # kernel = np.matrix('1; 1; 1')
        # matriceLigne = self.__fermeture(matriceLigne,kernel,minHeightFrame)

        matriceLigne, matriceCol = self.__set_border_to_black(matriceLigne,matriceCol,frame)

        # if frame == [(0, 8), (198, 59)]:
        #     cv2.imshow("Image1", matriceLigne)
        #     cv2.imshow("Image", matrice)
        #     cv2.waitKey(0)
        #     exit()

        noMoreOne = True
        pointLine = []
        for y in range(0,frame[1][1]+1 - frame[0][1]):
            if matriceLigne[y][0] > 0:
                if noMoreOne:
                    noMoreOne = False
                    pointLine.append( y-1 + frame[0][1])
                    pass
                pass
            elif not noMoreOne:
                noMoreOne = True
                pointLine.append( y + frame[0][1])


        noMoreOne = True
        pointCol = []
        for x in range(0,frame[1][0]+1 - frame[0][0]):
            if matriceCol[0][x] > 0:
                if noMoreOne:
                    noMoreOne = False
                    pointCol.append( x-1 + frame[0][0])
                    pass
                pass
            elif not noMoreOne:
                noMoreOne = True
                pointCol.append( x + frame[0][0])

        if pointCol == [] or pointLine == []:
            return [frame]


        # if pointCol == []:
        #     pointCol = [frame[0][0] , frame[1][0]]
        # elif pointLine == []:
        #     pointLine = [frame[0][1] , frame[1][1]]

        dichoArray = []
        for y in range(0,len(pointLine)/2):
            for x in range(0,len(pointCol)/2):
                dichoArray.append( [(pointCol[2*x],pointLine[2*y]),(pointCol[2*x+1],pointLine[2*y+1])] )
        pass



        ans = []
        for x in range(0,len(dichoArray)):
            ans += self.__paris_dichotomie_recursive(matriceOrigine, deep-1, dichoArray[x],thresholdWidth)

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

        # on va chercher toutes les lettres de la page
        dichoArray = self.new_leuven_dichotomie(image, cv2.bitwise_not(matrice), deep=10)
        thresholdWidth = 0
        # minHeightFrame = 1000000
        for x in range(0,len(dichoArray)):
            # minWidthFrame = min(minWidthFrame, dichoArray[x][1][0] - dichoArray[x][0][0] )
            thresholdWidth += dichoArray[x][1][0] - dichoArray[x][0][0]
        thresholdWidth /= len(dichoArray)



        dichoArray = self.__paris_dichotomie_recursive(matrice,10,frame,thresholdWidth)

        for x in range(0,len(dichoArray)):
            cv2.rectangle(matrice, dichoArray[x][0], dichoArray[x][1], (125, 0, 0), 1)


        # cv2.imshow("Image", matrice)
        # cv2.waitKey(0)
        return dichoArray





def main():
    C = Content()
    pass

if __name__ == '__main__':
    main()
