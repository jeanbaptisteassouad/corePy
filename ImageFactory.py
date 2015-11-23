import pickle
import cv2
import numpy as np


class ImageFactory(object):
    """docstring for ImageFactory"""


    def __init__(self, path, feature_extractor, content_extractor):
        super(ImageFactory, self).__init__()
        self.path = path
        self.image = cv2.imread(self.path)
        self.image_display = self.image.copy()
        self.gray = self.__prepare_image_gray()
        self.gray_display = self.gray.copy()
        self.feature_extractor = feature_extractor
        self.content_extractor = content_extractor
        self.list_content = content_extractor(self.image,self.gray)
        self.list_feature = np.array([])
        self.list_classes = np.array([])

    def extract_feature_for_all_content(self):
        self.list_feature = np.zeros( (len(self.list_content),500) )
        for x in range(0,len(self.list_content)):
            content_image = self.image[self.list_content[x][0][1]:self.list_content[x][1][1]+1, self.list_content[x][0][0]:self.list_content[x][1][0]+1]
            content_gray = self.gray[self.list_content[x][0][1]:self.list_content[x][1][1]+1, self.list_content[x][0][0]:self.list_content[x][1][0]+1]
            self.list_feature[x] = self.feature_extractor(content_image,content_gray)
            pass
        pass

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
        for x in range(0,len(self.list_classes)):
            if self.list_classes[x][1] == 1:
                self.image_display[self.list_content[x][0][1]:self.list_content[x][1][1]+1, self.list_content[x][0][0]:self.list_content[x][1][0]+1 , 0] += 100
        for x in range(0,len(self.list_content)):
            cv2.rectangle(self.image_display, self.list_content[x][0], self.list_content[x][1], (255, 0, 0), 1)



def main():
    F = Feature()
    C = Content()
    I = ImageFactory("core/test.png",F.projection_histogram,C.new_leuven_dichotomie)
    I.drawing_image_cv2()
    cv2.imshow("Image", I.image_display)
    cv2.waitKey(0)
    pass

if __name__ == '__main__':
    from Feature import Feature
    from Content import Content
    main()


