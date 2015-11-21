import numpy as np
import multiprocessing as mp

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


def main():
    C = Content()
    pass

if __name__ == '__main__':
    main()
