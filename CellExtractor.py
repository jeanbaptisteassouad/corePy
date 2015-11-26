import cv2
import numpy as np


class Tree(object):
    def __init__(self, data, pos, depth, start_line):
        self.data = data
        self.position = pos
        self.start_line = start_line
        self.depth = depth
        self.forest = []

    @staticmethod
    def __in_interval(a,b,c):
        if b <= a <= c:
            return True
        else:
            return False
        pass

    def intersection(self,b):
        inter = [0,0]
        if Tree.__in_interval(self.data[0], b[0], b[1]):
            inter[0] = self.data[0]
        elif Tree.__in_interval(b[0], self.data[0], self.data[1]):
            inter[0] = b[0]
        if Tree.__in_interval(self.data[1], b[0], b[1]):
            inter[1] = self.data[1]
        elif Tree.__in_interval(b[1], self.data[0], self.data[1]):
            inter[1] = b[1]
        return inter

    def intersect(self,tree):
        if self.intersection(tree.data) != [0,0]:
            return True
        else:
            return False

    def grow_forest(self, datas):
        for i in range(len(datas)):
            _intersection = self.intersection(datas[i][0])
            if _intersection != [0,0]:
                self.forest.append(Tree(_intersection,[self.position[0],datas[i][1][1]],self.depth+1,self.start_line))
                pass
        pass

    def is_leaf(self):
        if self.forest == []:
            return True
        else:
            return False

    def return_leafs(self,leaf_list):
        if not self.is_leaf():
            for son in self.forest:
                son.return_leafs(leaf_list)
        else:
            if self.depth > 1:
                leaf_list.append(self)
        pass

    @staticmethod
    def recursive(tree, layout, line_number):
        length = len(layout)
        if line_number >= length:
            return 0
        else:
            tree.grow_forest(layout[line_number][1])
            for son in tree.forest:
                Tree.recursive(son, layout, line_number+1)


class CellExtractor(object):
    def __init__(self):
        pass

    @staticmethod
    def preprocess(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
        img = cv2.Laplacian(img, cv2.CV_8U)
        # dilate and erode - helps getting a clearer layout view of the image
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        img = cv2.erode(cv2.dilate(img, element), element)
        # Return edge image
        return img

    @staticmethod
    def skeleton(img):
        size = np.size(img)
        skel = np.zeros(img.shape,np.uint8)

        ret,img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        done = False

        while( not done):
            eroded = cv2.erode(img,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = eroded.copy()

            zeros = size - cv2.countNonZero(img)
            if zeros==size:
                done = True
        return skel

    @staticmethod
    def local_projection(img):
        h, w = img.shape
        x = np.zeros(h)
        y = np.zeros(w)
        for i in range(h):
            for j in range(w):
                if img[i][j] == 255:
                    x[i] += 1
                    y[j] += 1
        return x, y

    @staticmethod
    def remove_lines(x, mean_percentile, max_percentile):
        mean, max = (mean_percentile*np.mean(x),max_percentile*np.max(x))
        # remove lines under mean and above maximal value
        for i in range(len(x)):
            if x[i] < mean or x[i] > max:
                x[i] = 0
        return x

    @staticmethod
    def __ouverture(matrice,kernel, itera):
        matriceTmp = cv2.erode(matrice, kernel, iterations=itera)
        return cv2.dilate(matriceTmp, kernel, iterations=itera)

    @staticmethod
    def __fermeture(matrice,kernel, itera):
        matriceTmp = cv2.dilate(matrice, kernel, iterations=itera)
        return cv2.erode(matriceTmp, kernel, iterations=itera)

    @staticmethod
    def remove_lines_2(img):
        print(img.shape)
        height, width, channels = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, matrice = cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
        matrice = cv2.Laplacian(matrice, cv2.CV_8U)

        kernel = np.matrix('1; 1; 1')
        matriceVerti = CellExtractor.__ouverture(matrice,kernel,20)
        kernel = np.matrix('1 1 1')
        matriceVerti = CellExtractor.__fermeture(matriceVerti,kernel,1)

        kernel = np.matrix('1 1 1')
        matriceHori = CellExtractor.__ouverture(matrice,kernel,20)
        kernel = np.matrix('1; 1; 1')
        matriceHori = CellExtractor.__fermeture(matriceHori,kernel,1)

        for y in range(0,height):
            for x in range(0,width):
                if x == 0 or x == width-1 or y == 0 or y == height-1:
                    matrice[y][x] = 0
                else:
                    matrice[y][x] -= min(int(matriceVerti[y][x]) + int(matriceHori[y][x]),255)
                pass
            pass
        retval , matrice = cv2.threshold(matrice,125,255,cv2.THRESH_BINARY)
        return matrice

    @staticmethod
    def clear_thin(layout, seuil):
        start, stop = 0, 0
        for i in range(len(layout)):
            if layout[i] != 0 and start == 0:
                start = i
            if layout[i] == 0 and start != 0:
                stop = i
            val = (stop - start)
            if val > 0:
                if val <= seuil:
                    for j in range(start,stop):
                        layout[j] = 0
                start, stop = 0, 0
        return layout

    @staticmethod
    def clean_img(img, x, y):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if x[i] == 0 or y[j] == 0:
                    img[i][j] = 0
        return img

    @staticmethod
    def find_empty_spaces(layout, seuil):
        start, stop, res = 0, 0, []
        for i in range(len(layout)):
            if layout[i] != 0 and start != 0:
                stop = i
            if layout[i] == 0 and start == 0:
                start = i
            val = (stop - start)
            if val > 0:
                if val >= seuil:
                    res.append((start,stop))
                start, stop = 0, 0
        # Checks for possible empty spaces at the right extremity of the image
        if len(layout) - start > 0 and len(layout) - start >= seuil:
            res.append((start, len(layout)))
        return res

    # works with an array of tuple [(val1, val2) ... ]
    @staticmethod
    def threshold_spacing(vector):
        var, mean = [], 0
        for i in range(len(vector)):
            var.append(vector[i][1] - vector[i][0])
            mean += vector[i][1] - vector[i][0]
        retval = np.sqrt(np.var(var))
        retval = np.sqrt(mean / len(vector))
        return retval

    # works with an array of tuple [(val1,val2) ... ]
    @staticmethod
    def threshold(vector, thr, thresh):
        # returns array of tuple [(val1,val2), ...]
        for i in range(len(vector)):
            if vector[i][1] - vector[i][0] > thresh:
                thr.append(True)
            else:
                thr.append(False)

    @staticmethod
    def create_layout(img):
        x, y = CellExtractor.local_projection(img)
        return CellExtractor.fill_layout(img, x, 0), CellExtractor.fill_layout(img, y, 1)

    @staticmethod
    def fill_layout(img, projection, direction):
        layout, spacing = [], CellExtractor.find_empty_spaces(projection, 1)
        thresh = []
        mean = CellExtractor.threshold_spacing(spacing)
        CellExtractor.threshold(spacing, thresh, mean)
        # left margin
        layout.append((spacing[0], []))
        for i in range(1,len(spacing)):
            if thresh[i]:
                layout.append((spacing[i], []))
        # right margin
        layout.append((spacing[-1], []))
        for i in range(len(layout)-1):
            line_one = (layout[i][0][0] + layout[i][0][1])/2
            if direction == 0:
                line_two = (layout[i+1][0][0] + layout[i+1][0][1])/2
                local_x, local_y = CellExtractor.local_projection(img[line_one:line_two, :])
                #show(img[line_one:line_two, :])
                local_projection = local_y
                pass
            else:
                line_two = (layout[i+1][0][0] + layout[i+1][0][1])/2
                local_x, local_y = CellExtractor.local_projection(img[:, line_one:line_two])
                local_projection = local_x
                pass
            CellExtractor.fill_spacing_arrays(local_projection, layout[i], line_one, line_two)
        return layout

    @staticmethod
    def fill_spacing_arrays(projection, vector, line_one, line_two):
        white_spacing = CellExtractor.find_empty_spaces(projection, 1)
        if len(white_spacing) == 0:
            return
        mean = CellExtractor.threshold_spacing(white_spacing)
        # threshold white spaces :
        thresh = []
        # Add first white space (usually corresponds to the left margin
        #vector[1].append((white_spacing[0], [line_one, line_two]))
        CellExtractor.threshold(white_spacing, thresh, mean)
        for i in range(1, len(white_spacing)):
            if thresh[i]:
                vector[1].append((white_spacing[i],[line_one, line_two]))
        # add right margin
        #vector[1].append((white_spacing[-1], [line_one, line_two]))


def show(img):
    cv2.imshow("image",img)
    cv2.waitKey()


def _find(layout,starting_line):
    if not layout:
        return []
    tree = [Tree(layout[starting_line][1][i][0], layout[starting_line][1][i][1], 0, starting_line) for i in range(len(layout[starting_line][1]))]
    leaf_list = []
    for i in range(len(tree)):
        Tree.recursive(tree[i], layout, 1)
        tree[i].return_leafs(leaf_list)
    return leaf_list
    pass


def _find_all(layout):
    leafs = []
    for i in range(len(layout)):
        leafs+=_find(layout,i)
    # clean leaf list
    terminate, index = False, 0
    while not terminate:
        terminate = True
        for i in range(index+1,len(leafs)):
            if leafs[index].intersect(leafs[i]):
                leafs[i].depth = -1
                terminate = False
                pass
        index+=1
        pass
    retval = []
    for leaf in leafs:
        if leaf.depth != -1:
            retval.append(leaf)
    return retval


def _draw(leaf_list, img, direction):
    for leaf in leaf_list:
        #print("line depth : "+str(leaf.depth)+", interval : "+str(leaf.data)+", pos : "+str(leaf.position))
        if leaf.depth <= 1:
            continue
        line = (leaf.data[0] + leaf.data[1])/2
        if direction == 0:
            img[leaf.position[0]:leaf.position[1],line] = 255
            pass
        else:
            img[line,leaf.position[0]:leaf.position[1]] = 255
            pass
    pass

if __name__ == '__main__':
    # Trials
    raw = cv2.imread("core/table2.png")
    #img = CellExtractor.preprocess(raw)
    # Remove lines here
    #x, y = CellExtractor.local_projection(img)
    #x, y = CellExtractor.remove_lines(x, 0.3, 0.8), CellExtractor.remove_lines(y, 0.3, 0.8)
    #x, y = CellExtractor.clear_thin(x, 5), CellExtractor.clear_thin(y, 5)
    #img = CellExtractor.clean_img(img, x, y)

    #img = CellExtractor.skeleton(img)

    img = CellExtractor.remove_lines_2(raw)

    show(img)

    layout_x, layout_y = CellExtractor.create_layout(img)

    x_lines = _find_all(layout_x)
    y_lines = _find_all(layout_y)

    _draw(x_lines,img,0)
    _draw(y_lines,img,1)

    cv2.imwrite("output.png",img)
    show(img)

    pass