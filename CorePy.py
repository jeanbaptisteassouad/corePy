import numpy as np
import cv2

class CorePy(object):
    """docstring for CorePy"""
    def __init__(self, path, predictor):
        super(CorePy, self).__init__()
        self.image_current = None
        self.path = path
        self.predictor = predictor
        self.max_distance_kppv = 0

    def training_predict_classes_current(self):
        prediction_classes = np.zeros( (len(self.image_current.list_feature),2) )
        tot = 0
        for x in range(0,len(self.image_current.list_feature)):
            prediction_classes[x] , distance = self.predictor.predict( self.image_current.list_feature[x] )
            tot += prediction_classes[x]
            if distance >= 0:
                self.max_distance_kppv = max(self.max_distance_kppv,distance)
        print self.max_distance_kppv
        print tot
        self.image_current.list_classes = prediction_classes

    def training_predictor(self):
        self.predictor.train(self.image_current.list_feature , self.image_current.list_classes)
        pass








    # A refaire, provisoire
    def isMouseInsideFrame(self,frame,x,y):
        if x < frame[0][0] or y < frame[0][1]:
            return False
        if x > frame[1][0] or y > frame[1][1]:
            return False
        return True

    def mouseIsInAPotentialRegionOfInterest(self,x,y):
        for i in range(0,len(self.image_current.list_content)):
            if self.isMouseInsideFrame(self.image_current.list_content[i],x,y):
                return True, i
                pass
            pass
        return False, -1

    def selectRegionOfInterest(self,event, x, y, flags, param):
        # grab references to the global variables

        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            pass

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            boolean, indice = self.mouseIsInAPotentialRegionOfInterest(x,y)
            if boolean:
                if self.image_current.list_classes[indice][0] == 0:
                    self.image_current.list_classes[indice][0] = 1
                    self.image_current.list_classes[indice][1] = 0
                else:
                    self.image_current.list_classes[indice][0] = 0
                    self.image_current.list_classes[indice][1] = 1
                self.image_current.drawing_image_cv2()
                cv2.imshow('image', self.image_current.image_display)
                pass


        if event == cv2.EVENT_MOUSEMOVE:
            pass

    def dummyCallBack(self,event, x, y, flags, param):
        pass


def functionTest(path,Core,F,C):
    print "SWAG"
    Core.image_current = ImageFactory(path,F.projection_histogram,C.new_leuven_dichotomie)
    Core.image_current.extract_feature_for_all_content()
    Core.training_predict_classes_current()
    Core.image_current.drawing_image_cv2()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', Core.selectRegionOfInterest)
    cv2.imshow('image', Core.image_current.image_display)
    cv2.waitKey(0)
    cv2.setMouseCallback('image', Core.dummyCallBack)
    Core.training_predictor()
    pass


def main():
    F = Feature()
    C = Content()
    # M = Mlp()
    K = Kppv()
    Core = CorePy("",K)

    # Training
    functionTest("core/test.png",Core,F,C)
    functionTest("core/hpc1.png",Core,F,C)
    functionTest("core/hpc2.png",Core,F,C)
    functionTest("core/hpc3.png",Core,F,C)
    functionTest("core/hpc4.png",Core,F,C)
    functionTest("core/bad.png",Core,F,C)
    functionTest("core/hpc5.png",Core,F,C)
    functionTest("core/hpc6.png",Core,F,C)
    functionTest("core/hpc7.png",Core,F,C)
    functionTest("core/hpc8.png",Core,F,C)
    functionTest("core/bad3.png",Core,F,C)
    functionTest("core/hpc9.png",Core,F,C)
    functionTest("core/hpc10.png",Core,F,C)
    functionTest("core/hpc11.png",Core,F,C)
    functionTest("core/hpc12.png",Core,F,C)
    functionTest("core/hpc13.png",Core,F,C)
    functionTest("core/hpc14.png",Core,F,C)


    Core.predictor.serialize()
    Core.predictor.deserialize()

    # Find table and Check

    pass

if __name__ == '__main__':
    # from Mlp import Mlp
    from Kppv import Kppv
    from ImageFactory import ImageFactory
    from Feature import Feature
    from Content import Content
    main()

