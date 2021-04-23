import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import render as rr
from threeD_utilsv2 import threetool

class PRNetfacetool:
    def __init__(self):
        self.tduv2t = threetool()
    
    def IS_face(self, img):
        IS_detefaced = self.tduv2t.deteface(img)
        if len(IS_detefaced) == 0 :
            return False
        else:
            return True

    def PRNetdrawrect(self, img):
        
        IS_detefaced = self.tduv2t.deteface(img)
        if len(IS_detefaced) == 0 :
            return img
        else:
            pos = self.tduv2t.pre_v2(img)
            face_kps = self.tduv2t.face_kps(pos)
            face_kps_arry = []
            for i in range(face_kps.shape[0]):
                x = face_kps[i, 0]
                y = face_kps[i, 1]
                face_kps_arry.append((x, y))
            points = np.array(face_kps_arry, np.int32)
            x, y, w, h = cv2.boundingRect(points)
            face = cv2.rectangle(img.copy(), (x, y), (x+w,y+h), (0, 255, 0), 2)
            return face


        
    def blendImages(self, src, dst, mask, featherAmount=0.2):
        #indeksy nie czarnych pikseli maski
        maskIndices = np.where(mask != 0)
        #te same indeksy tylko, ze teraz w jednej macierzy, gdzie kazdy wiersz to jeden piksel (x, y)
        maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
        faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
        featherAmount = featherAmount * np.max(faceSize)

        hull = cv2.convexHull(maskPts)
        dists = np.zeros(maskPts.shape[0])
        for i in range(maskPts.shape[0]):
            dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

        weights = np.clip(dists / featherAmount, 0, 1)

        composedImg = np.copy(dst)
        composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

        return composedImg
###############################################################
    def PRNetvideo(self, img, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        width, height = width, height
        n = 1
        while height > 1080 or width > 1080 :
            n = n-0.1
            if n < 0:
                break
            height = int (height*n)
            width = int (width*n)
       
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outPRNet = cv2.VideoWriter('./videos/PRNetvideo.mov',fourcc, 20.0,(width, height))

        h = int (img.shape[0]*0.2)
        w = int (img.shape[1]*0.2)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        print("start")
        while True:
            _, frame = cap.read()
            if not _ :
                break
            frame = cv2.resize(frame, (width, height),interpolation=cv2.INTER_CUBIC)
            print("frameshape",frame.shape)
            print("imgshape",img.shape)
            cv2.imwrite("./Result_images/bugimg.jpg",frame)
            if  self.IS_face(frame) == False:
                outPRNet.write(frame)
                print("No swap")
            else:
                frame = self.PRNetfaceswap(frame, img)
                # frame = self.PRNetdrawrect(frame)
                outPRNet.write(frame)
                print("face")
            cv2.imshow("resframe", frame)
            

            key = cv2.waitKey(1)
            if key == 27:
                cap.release()
                cv2.destrouAllWindows()
                break

        print("Face Swap Finished")
###############################################################
    def PRNetfaceswap(self, img, ref_img):
        # first person
        # img = cv2.imread("./images/photo_test.jpg")
        imgdtype = img.dtype
        # height = int (img.shape[0]*0.2)
        # width = int (img.shape[1]*0.2)
        # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        [h, w, _] = img.shape
        # 3d reconstruction -> get texture
        pos, vertices,center = self.tduv2t.pre_v3_posandvertice(img)
        img = img/255
        texture = self.tduv2t.gettexture(img,pos)
        center = center.astype(int)
        center =tuple(center)
        # cv2.imshow("texture_1", texture)
        # cv2.waitKey(0)

        # Second person

        # ref_img = cv2.imread("./images/face04.jpg")
        ref_pos , ref_vertices, ref_center = self.tduv2t.pre_v3_posandvertice(ref_img) 
        ref_img = ref_img/255.
        ref_texture = self.tduv2t.gettexture(ref_img, ref_pos)

        # cv2.imshow("texture_2", ref_texture)
        # cv2.waitKey(0)
        # load eye mask
        uv_face_eye = cv2.imread('./Data/uv-data/uv_face_eyes.png',cv2.IMREAD_GRAYSCALE)/255
        uv_face = cv2.imread('./Data/uv-data/uv_face.png', cv2.IMREAD_GRAYSCALE)/255
        eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)
        # new_texture
        new_texture = self.blendImages(ref_texture*255,texture*255,eye_mask)/255.0

        ##
        triangles = self.tduv2t.gettriangles()

        vis_colors = np.ones((vertices.shape[0], 1))
        face_mask = rr.render_texture(vertices.T, vis_colors.T, triangles.T, h, w, c = 1)
        face_mask = np.squeeze(face_mask > 0).astype(np.float32)
        facemask_test = face_mask[:,:,np.newaxis]


        new_colors = self.tduv2t.get_colors_from_texture(new_texture)
        print(vertices.T.shape)
        print(triangles.T.shape)
        print(vis_colors.T.shape)
        new_img = rr.render_texture(vertices.T, new_colors.T, triangles.T, h, w, c = 3)

        print(vertices.T.shape)
        print(new_colors.T.shape)
        print(triangles.T.shape)
        new_image = img*(1 - face_mask[:,:,np.newaxis]) + new_img*face_mask[:,:,np.newaxis]

        print("facemask_test dtype", facemask_test.dtype)
        facemask_test = facemask_test*255
        facemask_test= facemask_test.astype(imgdtype)
        new_image = new_image*255
        new_image = new_image.astype(imgdtype)
        img = img*255
        img = img.astype(imgdtype)
        # SeamlessCloneimg = cv2.seamlessClone(new_image, img, facemask_test, center, cv2.NORMAL_CLONE )
        # return SeamlessCloneimg
        return new_image


if __name__ == '__main__':
    img = cv2.imread("./images/photo_test.jpg")
    ref_img = cv2.imread("./images/face04.jpg")
    faceswap = PRNetfacetool()
    cv2.imshow('result', faceswap.PRNetfaceswap(img, ref_img))
    cv2.waitKey(0)
    