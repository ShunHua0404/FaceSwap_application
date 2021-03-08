import cv2 
from threeD_utilsv2 import threetool
import numpy as np
from cartoon import Photo2Cartoon

video_path = "./videos/video_03.mp4"

cap = cv2.VideoCapture(video_path)

width = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.5)
height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.5)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./videos/facedete.mov',fourcc, 20.0,(width, height))
outcartoon = cv2.VideoWriter('./videos/cartoonvido.mov',fourcc, 20.0,(256, 256))


framecount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
count = 0
FPS = cap.get(cv2.CAP_PROP_FPS)

c2p = Photo2Cartoon()
tduv2t = threetool()
print(cap.isOpened())


while True:
    _, frame = cap.read()


    # image_info = frame.shape
    # height = image_info[0]
    # width = image_info[1]
    # dst_height = int(height * 0.5)
    # dst_width = int(width * 0.5)

    
   
    resframe = cv2.resize(frame, (width,height))
    smallframe = resframe 
    faceframe = cv2.resize(frame, (width,height))
   


    if (count%10 == 0 or count == 0):
        imgg = cv2.cvtColor(resframe, cv2.COLOR_BGR2RGB)
        cartoon = c2p.inference(imgg)
        if cartoon is None:
            print("None")
        else:
            outcartoon.write(cartoon)
            cv2.imshow("dd", cartoon)
    
    
    facepoint = tduv2t.deteface(faceframe)
    print(facepoint)
    
    
    if len(facepoint) == 0 :
        print("None")
    else:
        print("face")
        for point in facepoint:
            x1 = point[0]
            y1 = point[1]
            x2 = point[0] + point[2]
            y2 = point[1] + point[3]
            cv2.rectangle(faceframe, (x1, y1), (x2, y2), (0, 255, 0), 3)
        out.write(faceframe)
    count = count + 1
    cv2.imshow("frames", faceframe)
    cv2.imshow("123", resframe)

    
    key = cv2.waitKey(1)
    if key == 27:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        break
