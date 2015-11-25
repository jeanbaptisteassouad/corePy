import pickle
import cv2
import numpy as np


class TableFactory(object):
    """docstring for TableFactory"""
    def __init__(self, path,content_extractor):
        super(TableFactory, self).__init__()
        self.path = path
        self.image = cv2.imread(self.path)
        self.image_display = self.image.copy()
        self.gray = self.__prepare_image_gray()
        self.gray_display = self.gray.copy()
        self.content_extractor = content_extractor
        self.list_content = content_extractor(self.image,self.gray)



    def __prepare_image_gray(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        height,width,channel = self.image.shape
        for x in range(0,width):
            gray[0][x] = 255
            gray[height-1][x] = 255
        for y in range(0,height):
            gray[y][0] = 255
            gray[y][width-1] = 255
        for y in range(0,height):
            for x in range(0,width):
                if gray[y][x] >= 250:
                    gray[y][x] = 255
                    pass
                pass
            pass
        return gray


    def serialize(self):
        pass

    def deserialize(self):
        pass

    def drawing_image_cv2(self):
        self.image_display = self.image.copy()
        for x in range(0,len(self.list_content)):
            cv2.rectangle(self.image_display, self.list_content[x][0], self.list_content[x][1], (255, 0, 0), 1)



def main():
    C = Content()
    T = TableFactory("core/table4.png",C.paris_dichotomie)
    T.drawing_image_cv2()
    cv2.imshow("Image", T.image_display)
    cv2.waitKey(0)
    pass


if __name__ == '__main__':
    from Content import Content
    main()