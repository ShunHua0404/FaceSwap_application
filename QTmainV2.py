import sys
import cv2
from PyQt5 import uic, QtWidgets, QtGui, QtMultimedia, QtMultimediaWidgets
from PyQt5 import QtCore
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
from PRNet_faceswap_tool import PRNetfacetool
from cartoon import Photo2Cartoon
import numpy as np

PR = PRNetfacetool()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        uic.loadUi('./Qtuic/main.ui', self)
        self.FaceImgWindow = FaceSwap()
        self.FaceVideoWindow = FaceSwapVideo()
        self.pushButton_imgswap.clicked.connect(self.FaceImgWindow.show)
        self.pushButton_videoswap.clicked.connect(self.FaceVideoWindow.show)

class FaceSwapVideo(QtWidgets.QMainWindow):        
    def __init__(self):
        super(FaceSwapVideo, self).__init__()
        uic.loadUi('./Qtuic/faceswapVideo.ui', self)
        self.update()
        # self.mediaPlayer = QMediaPlayer(self)
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QtMultimediaWidgets.QVideoWidget()
        # self.videoWidget.setGeometry(self.pos().x(), self.pos().y(), self.width(), self.height())
        # self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.vbox = QtWidgets.QVBoxLayout(self)
        self.vbox.addWidget(self.videoWidget)
        self.video_widget.setLayout(self.vbox)
        self.mediaPlayer.setVideoOutput(self.videoWidget)
        # self.video_widget.setVisible(False)

        self.pushButton_right.clicked.connect(self.getvideofile)
        self.pushButton_left.clicked.connect(self.getOriImg)
        self.pushButton_start.clicked.connect(self.faceswap_start)

        self.faceimg = None
        self.ori_video_path = None
    
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self, 'Window Close', 'Are you sure you want to close the window?', QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
            print('Window closed')
            self.mediaPlayer.stop()
            self.hide()
        else:
            event.ignore()
    

    def faceswap_start(self):
        if(np.all(self.faceimg) != None and self.ori_video_path != None):
            PR.PRNetvideo(self.faceimg, self.ori_video_path)            
        else:
            print("No")


    def getOriImg(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',"D:\\","Image files(*.jpg)")
        if fname:
            self.faceimg = cv2.imread(fname)
            img = self.faceimg.copy()
            h, w, c = img.shape
            n = 1
            while(h > 540 or w > 540): 
                n = n-0.1
                h = int (h*n)
                w = int (w*n)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            img = PR.PRNetdrawrect(img)
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.img_label.setPixmap(QtGui.QPixmap.fromImage(qImg))

        

    def getvideofile(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',"D:\\","Video files(*.mp4 *MOV)")
        if fname:
            self.ori_video_path = fname
            # # self.video_widget.setLayout(self.vbox)
            # # self.video_widget.setVisible(True)
            
            self.mediaPlayer.setMedia(QtMultimedia.QMediaContent(QtCore.QUrl.fromLocalFile(fname)))
            self.mediaPlayer.play()
            
        



class FaceSwap(QtWidgets.QMainWindow):
    graph_trigger = QtCore.pyqtSignal(object)
    graph_trigger2 = QtCore.pyqtSignal(object)
    def __init__(self):
        super(FaceSwap, self).__init__()
        uic.loadUi('./Qtuic/faceswap.ui', self)
        self.faceswap = FaceSwap_thread()
        self.ori_img = None
        self.ref_img = None
        self.turncartoon = Photo2Cartoon()
        self.pushButton_left.clicked.connect(self.getimgfile)
        self.pushButton_right.clicked.connect(self.getimgfile_rightbutton)
        self.pushButton_Faceswap.clicked.connect(self.send_sth)
        
        self.pushButton_left_down.clicked.connect(self.cartoonstyle_left)
        self.pushButton_right_down.clicked.connect(self.cartoonstyle_right)
        
        
        # self.actionOpenfile.triggered.connect(self.)

    def send_sth(self):
        # self.faceswap.start()
        self.graph_trigger.connect(self.faceswap.get_ori_img)
        self.graph_trigger2.connect(self.faceswap.get_ref_img)
        self.graph_trigger.emit(self.ori_img)
        self.graph_trigger2.emit(self.ref_img)
        self.faceswap.start()
        self.faceswap.regraph_trigger.connect(self.showfaceswapped_img)

    def showfaceswapped_img(self, array):
        result = array
        h, w, c = result.shape    
        bytesPerline = 3*w
        qImg = QtGui.QImage(result.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label_mid.setPixmap(QtGui.QPixmap.fromImage(qImg))
    
    def cartoonstyle_left(self):
        if np.all(self.ori_img) != None and PR.IS_face(self.ori_img):
            self.ori_img = cv2.cvtColor(self.ori_img, cv2.COLOR_BGR2RGB)
            self.ori_img = self.turncartoon.inference(self.ori_img)
            img = self.ori_img.copy()
            img = PR.PRNetdrawrect(img)
            h, w, c = img.shape
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_left.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_left_down.hide()
        else:
            print('no face')
    def cartoonstyle_right(self):
        if np.all(self.ref_img) != None and PR.IS_face(self.ref_img):
            self.ref_img = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2RGB)
            self.ref_img = self.turncartoon.inference(self.ref_img)
            img = self.ref_img.copy()
            img = PR.PRNetdrawrect(img)
            h, w, c = img.shape
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_right.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_right_down.hide()
        else:
            print('no face')

    def getimgfile_rightbutton(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',"D:\\","Image files(*.jpg)")
        # self.label_right.setPixmap(QtGui.QPixmap(fname))
        self.ref_img = cv2.imread(fname)
        if (fname):
            print(self.ref_img.shape)
            h, w, c = self.ref_img.shape
            while(h > 1080 or w > 1080): 
                h = int (h*0.5)
                w = int (w*0.5)
                self.ref_img = cv2.resize(self.ref_img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            img = PR.PRNetdrawrect(self.ref_img)
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_right.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_right_down.show()
    
    def getimgfile(self):
        fname,_=QtWidgets.QFileDialog.getOpenFileName(self,'打開文件',"D:\\","Image files(*.jpg)")
        # self.label_right.setPixmap(QtGui.QPixmap(fname))
        self.ori_img = cv2.imread(fname)
        if (fname):
            print(self.ori_img.shape)
            h, w, c = self.ori_img.shape
            while(h > 1080 or w > 1080): 
                h = int (h*0.5)
                w = int (w*0.5)
                self.ori_img = cv2.resize(self.ori_img, (w, h), interpolation=cv2.INTER_CUBIC)
            
            img = PR.PRNetdrawrect(self.ori_img)
            bytesPerline = 3*w
            qImg = QtGui.QImage(img.data, w, h, bytesPerline, QtGui.QImage.Format_RGB888).rgbSwapped()
            self.label_left.setPixmap(QtGui.QPixmap.fromImage(qImg))
            self.pushButton_left_down.show()

        # self.label_left.setScaledContents(True)

class FaceSwapVideo_thread(QtCore.QThread):
    def __init__(self):
        super(FaceSwapVideo_thread, self).__init__()

class FaceSwap_thread(QtCore.QThread):
    regraph_trigger = QtCore.pyqtSignal(object)
    def __init__(self):
        super(FaceSwap_thread, self).__init__()
        self.ori_img = None
        self.ref_img = None

    def run(self):
        self.faceswaps()
    
    

    def get_ori_img(self, array):
        self.ori_img = array

    def get_ref_img(self, array):
        self.ref_img = array
        
    def faceswaps(self):
        if(np.all(self.ori_img) != None and np.all(self.ref_img) != None):
            if PR.IS_face(self.ref_img)  and PR.IS_face(self.ori_img):
                result = PR.PRNetfaceswap(self.ref_img, self.ori_img)
                # h, w, c = result.shape
                # n = 1
                # while(h > 540 or w > 540): 
                #     n = n-0.1
                #     h = int (h*n)
                #     w = int (w*n)
                #     result = cv2.resize(result, (w, h), interpolation=cv2.INTER_CUBIC)
                self.regraph_trigger.emit(result)
            else:
                print("No face")
        else:
            print('no')


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())