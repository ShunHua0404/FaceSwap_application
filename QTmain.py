import sys
import cv2
from PyQt5 import uic, QtWidgets, QtGui
from PyQt5 import QtCore
from PRNet_faceswap_tool import PRNetfacetool
from cartoon import Photo2Cartoon
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('./Qtuic/main.ui', self)
        self.FaceImgWindow = FaceSwap()
        self.FaceVideoWindow = FaceSwapVideo()
        self.pushButton_imgswap.clicked.connect(self.FaceImgWindow.show)
        self.pushButton_videoswap.clicked.connect(self.FaceVideoWindow.show)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.right_menu)

    def right_menu(self, pos):
         menu = QtWidgets.QMenu()

         # Add menu options
         hello_option = menu.addAction('Hello World')
         goodbye_option = menu.addAction('GoodBye')
         exit_option = menu.addAction('Exit')

         # Menu option events
         hello_option.triggered.connect(lambda: print('Hello World'))
         goodbye_option.triggered.connect(lambda: print('Goodbye'))
         exit_option.triggered.connect(lambda: exit())

         # Position
         menu.exec_(self.mapToGlobal(pos))

class FaceSwapVideo(QtWidgets.QMainWindow):        
    def __init__(self):
        super(FaceSwapVideo, self).__init__()
        uic.loadUi('./Qtuic/faceswapVideo.ui', self)

class FaceSwap(QtWidgets.QMainWindow):
    def __init__(self):
        super(FaceSwap, self).__init__()
        uic.loadUi('./Qtuic/faceswap.ui', self)
        self.imgfaceswap_flag = True
        self.ori_img = None
        self.ref_img = None
        self.PRnface = PRNetfacetool()
        self.turncartoon = Photo2Cartoon()
        self.pushButton_left.clicked.connect(self.getimgfile)
        self.pushButton_right.clicked.connect(self.getimgfile_rightbutton)
        self.pushButton_Faceswap.clicked.connect(self.checking)
        
        self.pushButton_left_down.clicked.connect(self.cartoonstyle)
        self.pushButton_left_down.hide()
        
        # self.actionOpenfile.triggered.connect(self.)
    
    def cartoonstyle(self):
        if np.all(self.ori_img) != None and self.PRnface.IS_face(self.ori_img):
            self.ori_img = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2RGB)
            self.ori_img = self.turncartoon.inference(self.ori_img)
            img = self.ori_img.copy()
            img = self.PRnface.PRNetdrawrect(img)
            h, w, c = img.shape
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_left.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_left_down.hide()
        else:
            print('no face')

    def checking(self):
        if self.imgfaceswap_flag == False:
            print("ing")
        else:
            self.imgfaceswap_flag = False
            self.faceswap()
    def faceswap(self):
        if(np.all(self.ref_img) != None and np.all(self.ori_img) != None):
            if self.PRnface.IS_face(self.ref_img)  and self.PRnface.IS_face(self.ori_img):
                result = self.PRnface.PRNetfaceswap(self.ref_img, self.ori_img)
                h, w, c = result.shape
                n = 1
                while(h > 540 or w > 540): 
                    n = n-0.1
                    h = int (h*n)
                    w = int (w*n)
                    result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
                bytesPerline = 3*w
                qImg = QtGui.QImage(result.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
                self.label_mid.setPixmap(QtGui.QPixmap.fromImage(qImg))
            else:
                print("No face")
        else:
            print('no')
            # print(self.ref_img)
            # print(self.ori_img)
            # print(np.all(self.ref_img))
            # print(np.all(self.ori_img))
            # print(np.all(self.ref_img != 0))
            # print(np.all(self.ori_img != 0))
            # print(np.all(self.ref_img == 0))
            # print(np.all(self.ori_img == 0))
        self.imgfaceswap_flag = True


    def getimgfile_rightbutton(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',"D:\\","Image files(*.jpg *.gif)")
        # self.label_right.setPixmap(QtGui.QPixmap(fname))
        self.ref_img = cv2.imread(fname)
        if (fname):
            img = self.ref_img.copy()
            print(img.shape)
            h, w, c = img.shape
            n = 1
            while(h > 540 or w > 540): 
                n = n-0.1
                h = int (h*n)
                w = int (w*n)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            img = self.PRnface.PRNetdrawrect(img)
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_right.setPixmap(QtGui.QPixmap.fromImage(qImg))
    

    def getimgfile(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',"D:\\","Image files(*.jpg *.gif)")
        # self.label_right.setPixmap(QtGui.QPixmap(fname))
        self.ori_img = cv2.imread(fname)
        if (fname):
            img = self.ori_img.copy()
            print(img.shape)
            h, w, c = img.shape
            n = 1
            while(h > 540 or w > 540): 
                n = n-0.1
                h = int (h*n)
                w = int (w*n)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            img = self.PRnface.PRNetdrawrect(img)
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_left.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_left_down.show()
        # self.label_left.setScaledContents(True)
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())