import numpy as np
import os
from skimage.transform import estimate_transform, warp
import cv2
from cv2 import cv2 as cv2 
from predictor import PosPrediction
import matplotlib.pyplot as plt



class threetool:

    def __init__(self):
        self.cas = cv2.CascadeClassifier('./data1/haarcascade_frontalface_alt2.xml')
        self.pos_predictor = PosPrediction(256, 256)
        self.pos_predictor.restore('./Data/net-data/256_256_resfcn256_weight')
        self.uv_kpt_ind = np.loadtxt('./Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt('./Data/uv-data/triangles.txt').astype(np.int32)
        self.face_ind = np.loadtxt('./Data/uv-data/face_ind.txt').astype(np.int32)
    #
    def deteface(self, img):
        img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        faces = self.cas.detectMultiScale(img_gray,2,3,0,(30,30))
        
        # bbox = np.array([faces[0,0],faces[0,1],faces[0,0]+faces[0,2],faces[0,1]+faces[0,3]])
        # left = bbox[0]; top = bbox[1]; right = bbox[2]; bottom = bbox[3]
        # , left, top, right, bottom
        return faces
    #
    def cutimg(self, img):
        img_gray= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        faces = self.cas.detectMultiScale(img_gray,2,3,0,(30,30))
        bbox = np.array([faces[0,0],faces[0,1],faces[0,0]+faces[0,2],faces[0,1]+faces[0,3]])

        left = bbox[0]; top = bbox[1]; right = bbox[2]; bottom = bbox[3]
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size*1.6)

        src_pts = np.array([[center[0]-size/2, center[1]-size/2], 
                            [center[0] - size/2, center[1]+size/2], 
                            [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,255], [255, 0]]) #图像大小256*256
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        img = img/255.
        cropped_img = warp(img, tform.inverse, output_shape=(256, 256))

        return bbox, cropped_img

    def pre(self, img):
        pos = self.pos_predictor.predict(img)
        
        return pos

    def draw_kps(self, img, kps, point_size=2):
        img = np.array(img*255,np.uint8)
        for i in range(kps.shape[0]):
            cv2.circle(img,(int(kps[i,0]),int(kps[i,1])),point_size,(0,255,0),-1)
        return img

    def getuvkptind(self):
        
        uvkpt = self.uv_kpt_ind

        return uvkpt

    def face_kps(self, pos):
        facekps = pos[self.uv_kpt_ind[1,:],self.uv_kpt_ind[0,:],:]

        return facekps

    def gettexture(self, img, pos):
        
        texture = cv2.remap(img, pos[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))

        return texture
    def maptexture(self, pos, texture, img):
        all_colors = np.reshape(texture, [256*256, -1])
        colors = all_colors[self.face_ind, :]
        all_vertices = np.reshape(pos, [256*256, -1])
        vertices = all_vertices[self.face_ind, :]

        print(vertices.shape) # texutre每个像素对应的3D坐标
        print(self.triangles.shape) #每个三角网格对应的像素索引
        print(colors.shape) #每个三角形的颜色
        #tri_depth = (vertices[self.triangles[:,0],2 ] + vertices[self.triangles[:,1],2] + vertices[self.triangles[:,2],2])/3. 
        #获取三角形每个顶点的color，平均值作为三角形颜色
        tri_tex = (colors[self.triangles[:,0] ,:] + colors[self.triangles[:,1],:] + colors[self.triangles[:,2],:])/3.
        tri_tex = tri_tex*255
        
        img_3D = np.zeros_like(img,dtype=np.uint8)
        for i in range(self.triangles.shape[0]):
            cnt = np.array([(vertices[self.triangles[i,0],0],vertices[self.triangles[i,0],1]),
                (vertices[self.triangles[i,1],0],vertices[self.triangles[i,1],1]),
                (vertices[self.triangles[i,2],0],vertices[self.triangles[i,2],1])],dtype=np.int32)
            img_3D = cv2.drawContours(img_3D,[cnt],0,tri_tex[i],-1)

        return img_3D, vertices, tri_tex
        

    def angle2matrix(self, angles):
        x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
        # x
        Rx=np.array([[1,              0,                0],
                    [0, np.math.cos(x),  -np.math.sin(x)],
                    [0, np.math.sin(x),   np.math.cos(x)]])
        # y
        Ry=np.array([[ np.math.cos(y), 0, np.math.sin(y)],
                    [              0, 1,              0],
                    [-np.math.sin(y), 0, np.math.cos(y)]])
        # z
        Rz=np.array([[np.math.cos(z), -np.math.sin(z), 0],
                    [np.math.sin(z),  np.math.cos(z), 0],
                    [             0,               0, 1]])

        R=Rz.dot(Ry.dot(Rx))
        return R.astype(np.float32)
    
    def rotateimg(self, angle, vertices, tri_tex, img):

        rotated_vertices = vertices.dot(angle.T)

        # 把图像拉到画布上
        # ori_x = np.min(vertices[:,0])
        # ori_y = np.min(vertices[:,1])
        # rot_x = np.min(rotated_vertices[:,0])
        # rot_y = np.min(rotated_vertices[:,1])
        # shift_x = ori_x-rot_x
        # shift_y = ori_y-rot_y
        # rotated_vertices[:,0]=rotated_vertices[:,0]+shift_x
        # rotated_vertices[:,1]=rotated_vertices[:,1]+shift_y


        img_3D = np.zeros_like(img,dtype=np.uint8)
        mask = np.zeros_like(img,dtype=np.uint8)
        fill_area=0
        for i in range(self.triangles.shape[0]):
            cnt = np.array([(rotated_vertices[self.triangles[i,0],0],rotated_vertices[self.triangles[i,0],1]),
                (rotated_vertices[self.triangles[i,1],0],rotated_vertices[self.triangles[i,1],1]),
                (rotated_vertices[self.triangles[i,2],0],rotated_vertices[self.triangles[i,2],1])],dtype=np.int32)
            mask = cv2.drawContours(mask,[cnt],0,(255,255,255),-1)
            if(np.sum(mask[...,0])>fill_area):
                fill_area = np.sum(mask[...,0])
                img_3D = cv2.drawContours(img_3D,[cnt],0,tri_tex[i],-1)

        return img_3D
