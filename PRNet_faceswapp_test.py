import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import render as rr
from threeD_utilsv2 import threetool
tduv2t = threetool()
# np.set_printoptions(linewidth=50,edgeitems=100) 
def blendImages(src, dst, mask, featherAmount=0.2):
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
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    #indeksy nie czarnych pikseli maski
    maskIndices = np.where(mask != 0)
    #src[maskIndices[0], maskIndices[1]] zwraca piksele w nie czarnym obszarze maski

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


# first person
img = cv2.imread("./images/photo_test.jpg")
imgdtype = img.dtype
height = int (img.shape[0]*0.2)
width = int (img.shape[1]*0.2)
img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
# bbox ,cutimg_1 = tduv2t.cutimg(img)
[h, w, _] = img.shape
# 3d reconstruction -> get texture
pos, vertices,center = tduv2t.pre_v3_posandvertice(img)
img = img/255
texture = tduv2t.gettexture(img,pos)
center = center.astype(int)
center =tuple(center)
cv2.imshow("texture_1", texture)
cv2.waitKey(0)

# Second person

ref_img = cv2.imread("./images/test.jpg")
ref_pos , ref_vertices, ref_center = tduv2t.pre_v3_posandvertice(ref_img) 
ref_img = ref_img/255.
ref_texture = tduv2t.gettexture(ref_img, ref_pos)

cv2.imshow("texture_2", ref_texture)
cv2.waitKey(0)
# load eye mask
uv_face_eye = cv2.imread('./Data/uv-data/uv_face_eyes.png',cv2.IMREAD_GRAYSCALE)/255
uv_face = cv2.imread('./Data/uv-data/uv_face.png', cv2.IMREAD_GRAYSCALE)/255
eye_mask = (abs(uv_face_eye - uv_face) > 0).astype(np.float32)
# new_texture
new_texture = blendImages(ref_texture*255,texture*255,eye_mask)/255.0

##
triangles = tduv2t.gettriangles()

vis_colors = np.ones((vertices.shape[0], 1))
face_mask = rr.render_texture(vertices.T, vis_colors.T, triangles.T, h, w, c = 1)
face_mask = np.squeeze(face_mask > 0).astype(np.float32)
facemask_test = face_mask[:,:,np.newaxis]
# cv2.imshow("facemask_test",facemask_test)
# cv2.waitKey(0)


new_colors = tduv2t.get_colors_from_texture(new_texture)
print(vertices.T.shape)
print(triangles.T.shape)
print(vis_colors.T.shape)
new_img = rr.render_texture(vertices.T, new_colors.T, triangles.T, h, w, c = 3)
# cv2.imshow("new", new_img)
# cv2.waitKey(0)
print(vertices.T.shape)
print(new_colors.T.shape)
print(triangles.T.shape)

new_image = img*(1 - face_mask[:,:,np.newaxis]) + new_img*face_mask[:,:,np.newaxis]
print(new_image.shape)
# height = int (new_image.shape[0]*0.2)
# width = int (new_image.shape[1]*0.2)
# new_image = cv2.resize(new_image, (width, height), interpolation=cv2.INTER_CUBIC)

imagetest = img*(1 - face_mask[:,:,np.newaxis])
imagetests = new_img*face_mask[:,:,np.newaxis]
cv2.imshow("imagetest", imagetest)#without face
cv2.imshow("imagetests", imagetests)#face

cwidth, cheight = imagetests.shape[1], imagetests.shape[0]
# center = (int(cwidth//2), int(cheight/2))
image_mask = 255 * np.ones(imagetests.shape, imagetest.dtype).astype(np.float32)

cv2.imshow("image_mask", image_mask)#white
cv2.waitKey(0)
cv2.imshow("imagetests", imagetests)
cv2.imshow("img", img)
cv2.imshow("facemask_test", facemask_test)
print(new_image)
print(img)
print(facemask_test)
facemask_test = facemask_test*255
facemask_test= facemask_test.astype(imgdtype)
print(face_mask.dtype)
# img = img*255
# img = img.astype(imgdtype)
print("img", img.dtype)
# new_image = new_image*255
# new_image = new_image.astype(imgdtype)
mask = 255*np.ones(new_image.shape, imgdtype)
print(mask)
print("mask",mask.shape)
print("new_image", new_image.dtype)
print(facemask_test)
print("new_image", new_image.shape)
print("img", img.shape)
print("facemask_test", facemask_test.shape)
print("center",center)
cv2.waitKey(0)
SeamlessCloneimg = cv2.seamlessClone(new_image, img,facemask_test, center, cv2.NORMAL_CLONE)
# SeamlessCloneimage = cv2.resize(SeamlessCloneimg, (widthcd, height), interpolation=cv2.INTER_CUBIC)


# cv2.imshow("seam",SeamlessCloneimage)
cv2.imshow("uv_face_eye",uv_face_eye)
cv2.imshow("uv_face",uv_face)
cv2.imshow("eye_mask",eye_mask)
cv2.imshow("new_texture", new_texture)
cv2.imshow("new_image", new_image)
cv2.imwrite("./Result_images/new_images.jpg", new_image)
cv2.waitKey(0)


