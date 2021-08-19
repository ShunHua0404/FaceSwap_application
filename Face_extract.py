import cv2
from threeD_utilsv2 import threetool
import numpy as np


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index
# Load images
tduv2t = threetool()
face_path = "./images/photo_test.jpg"
face_path_2 = "./images/face04.jpg"

## Face_1 scale
img = cv2.imread(face_path)
# if img.shape[0] > 500 or img.shape[1] > 500:
#     height = int (img.shape[0]*0.2)
#     width = int (img.shape[1]*0.2)
#     img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img)
## Face_2 scale
img2 = cv2.imread(face_path_2)
# if img2.shape[0] > 1000 or img2.shape[1] > 1000:
#     height = int (img2.shape[0]*0.5)
#     width = int (img2.shape[1]*0.5)
#     img2 = cv2.resize(img2, (width, height), interpolation=cv2.INTER_CUBIC)
# Image to gray
img2_gray = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY)
img2_new_face = np.zeros_like(img2)
result_img = img2.copy()#
##Face_1 PRN pre pos
pos = tduv2t.pre_v3_posandvertice(img, get_vertices_center=False)
face_kps = tduv2t.face_kps(pos)#shape = (68,3)
face_kps_arry = []

for i in range(face_kps.shape[0]):
    x = face_kps[i, 0]
    y = face_kps[i, 1]

    face_kps_arry.append((x, y))

#Face_1
img_1 = img/255 #why(?) 
face_68kps_img = (tduv2t.draw_kps(img_1, face_kps))
points = np.array(face_kps_arry, np.int32)
face68points_convexhull = cv2.convexHull(points)
###########################################

#Face_2 PRN pre pos2
pos2 = tduv2t.pre_v3_posandvertice(img2, get_vertices_center=False)
face2_kps = tduv2t.face_kps(pos2)
face2_kps_arry = []

for i in range(face2_kps.shape[0]):
    x = face2_kps[i, 0]
    y = face2_kps[i, 1]

    face2_kps_arry.append((x, y))

#Face_2
img_2 = img2/255
face2_68kps_img = (tduv2t.draw_kps(img_2, face2_kps))
points2 = np.array(face2_kps_arry, np.int32)
face68points2_convexhull = cv2.convexHull(points2)


#########################################################
cv2.polylines(face_68kps_img, [face68points_convexhull], True, (0, 0, 255), 2)
cv2.fillConvexPoly(mask, face68points_convexhull, 255)
# face_mask_image = cv2.bitwise_and(face_68kps_img, face_68kps_img, mask = mask)


# Delaunay triangulation

rect = cv2.boundingRect(face68points_convexhull)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(face_kps_arry)
triangles = subdiv.getTriangleList()
triangles = np.array(triangles, dtype = np.int32)
indexes_triangles = []
for t in triangles:
    pt1 = (t[0], t[1])
    pt2 = (t[2], t[3])
    pt3 = (t[4], t[5])

    index_pt1 = np.where((points == pt1).all(axis=1))
    index_pt1 = extract_index_nparray(index_pt1)
    
    index_pt2 = np.where((points == pt2).all(axis=1))
    index_pt2 = extract_index_nparray(index_pt2)

    index_pt3 = np.where((points == pt3).all(axis=1))
    index_pt3 = extract_index_nparray(index_pt3)

    if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangles = [index_pt1, index_pt2, index_pt3]
        indexes_triangles.append(triangles)
    # print(index_pt1)

    # cv2.line(Delaunay_triangul_img, pt1,pt2, (0, 0,255))
    # cv2.line(Delaunay_triangul_img, pt2,pt3, (0, 0,255))
    # cv2.line(Delaunay_triangul_img, pt1,pt3, (0, 0,255))
# print(triangles)

lines_space_mask = np.zeros_like(img_gray)

#triangulation of both faces
for triangles_index in indexes_triangles:
    # Triangulation of first face
    tr1_pt1 = face_kps_arry[triangles_index[0]]
    tr1_pt2 = face_kps_arry[triangles_index[1]]
    tr1_pt3 = face_kps_arry[triangles_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = img[y: y + h, x: x+w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)
    
    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    cropped_triangle = cv2.bitwise_and(cropped_triangle, cropped_triangle, mask = cropped_tr1_mask)
    # cv2.line(img, tr1_pt1, tr1_pt2, (0, 0,255))
    # cv2.line(img, tr1_pt2, tr1_pt3, (0, 0,255))
    # cv2.line(img, tr1_pt1, tr1_pt3, (0, 0,255))

    # Triangulation of second face
    tr2_pt1 = face2_kps_arry[triangles_index[0]]
    tr2_pt2 = face2_kps_arry[triangles_index[1]]
    tr2_pt3 = face2_kps_arry[triangles_index[2]]

    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2
    cropped_tr2_mask = np.zeros((h, w), np.uint8)
    
    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)


    # cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0,255))
    # cv2.line(img2, tr2_pt2, tr2_pt3, (0, 0,255))
    # cv2.line(img2, tr2_pt1, tr2_pt3, (0, 0,255))
    

    # warp triangle
    points = np.float32(points)
    points2 = np.float32(points2)
    
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
    # cv2.imshow("warped_triangle1",warped_triangle)
    # Reconstruct destination face
    triangle_area = img2_new_face[y: y + h, x: x+w]
    triangle_area_gray = cv2.cvtColor(triangle_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(triangle_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
    triangle_area = cv2.add(triangle_area, warped_triangle)
    img2_new_face[y: y + h, x: x+w] = triangle_area
# Face swapped(putting 1st into  2nd)
# img2_new_face_gray = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2GRAY)
# _, background = cv2.threshold(img2_new_face_gray, 1, 255, cv2.THRESH_BINARY_INV)
# background = cv2.bitwise_and(img2, img2, mask=background)
# result = cv2.add(background, img2_new_face)
###############
img2_face_mask = np.zeros_like(img2_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, face68points2_convexhull, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)

img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
result = cv2.add(img2_head_noface, img2_new_face)

(x, y, w, h) = cv2.boundingRect(face68points2_convexhull)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
###################################
cv2.imshow("newface", img2_new_face)
cv2.imshow("image1",img)
cv2.imshow("image2",img2)
# cv2.imshow("result", result)
cv2.imshow("seamlessclone", seamlessclone)

# Save Result imgaes
# cv2.imwrite("./Result_images/newface.jpg", img2_new_face)
# cv2.imwrite("./Result_images/background.jpg", background)
# cv2.imwrite("./Result_images/image1.jpg",img)
# cv2.imwrite("./Result_images/image2.jpg",img2)
# cv2.imwrite("./Result_images/result.jpg", result)
# cv2.imwrite("./Result_images/seamlessclone.jpg", seamlessclone)

cv2.waitKey(0)
