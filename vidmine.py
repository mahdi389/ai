from ultralytics import YOLO
from PIL import Image
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os
  
writer = cv2.VideoWriter("outputvideo.mp4",cv2.VideoWriter_fourcc(*"MP4V"),30,(1920,1080))

things = []
class thing :
    def __init__(self, x1, y1, x2, y2, c, id):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.xx = int((x1 + x2) /2)
        self.yy = int((y1 + y2) /2)
        self.id = id
        self.c = c


blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

car_class = 2
motorcycle_class = 3
bus_class = 5
truck_class = 7
ourclasses = [2, 3, 5, 7]

model = YOLO("yolov8n.pt")
cap=cv2.VideoCapture("D:\heavyApps\yolov8\infer\s.mp4")



count = 0 #counting th e frames
things_counter = 0

def is_it_new(n , li):
    result = -1#it means its new
    for thingg in li:
        if n.c == thingg.c:
            if n.xx > thingg.x1 and n.xx < thingg.x2:
                if n.yy > thingg.y1 and n.yy < thingg.y2:
                    result = thingg.id
    #print(f'haaaaaaaaaaaaa{result}')
    return result



frames = []

while True:

    greens = 0
    reds = 0


    ret,frame = cap.read()
    if ret == False:
        break
    else:
        count = count+1
    
    #print(frame.shape)
    #print(ret)
    #cv2.imshow('Frame',frame)
    results = model.predict(source=frame)  #, save=True, save_txt=True, project = f"frame{count}" 
    #print(results)
    a = results[0].boxes.data
    px=pd.DataFrame(a.cpu()).astype("float")

    for index,row in px.iterrows():
        x1=int(row[0])#buttom left
        y1=int(row[1])
        x2=int(row[2])#top right
        y2=int(row[3])
        percentage = 100 * int(row[4])
        thingclass = int(row[5])


        if thingclass in ourclasses :
            n = thing(x1, y1, x2, y2, thingclass, -1)
            result = is_it_new(n, things)
            if result == -1:
                things_counter = things_counter + 1
                n.id = things_counter
                things.append(n)
                

            else:
                n.id = things[result-1].id
                things[result-1] = n
            
            xx = (x1 + x2) /2
            yy = (y1 + y2) /2
            if xx > (1920 / 4) and xx < 3*(1920 / 4):
                if yy < 3 *(1080 / 4 ) and yy > 2*(1080 / 4):
                    cv2.rectangle(frame,(x1,y1),(x2,y2),red,2)
                    cv2.putText(frame,str(n.id),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,blue,1)
                    reds +=1
                    if result == -1:
                        croped_obj = frame[y1:y2, x1:x2]
                        cv2.imwrite(f"ibj_{n.id}.jpg", croped_obj)
                if yy < 2 *(1080 / 4 ) and yy > (1080 / 4):
                    cv2.rectangle(frame,(x1,y1),(x2,y2),green,2)
                    cv2.putText(frame,str(n.id),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,blue,1)
                    greens +=1
                    if result == -1:
                        croped_obj = frame[y1:y2, x1:x2]
                        cv2.imwrite(f"ibj_{n.id}.jpg", croped_obj)

        
    ##########################print the line on the top
  
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    org = (50, 50) 

    text = f"greens = {greens} and reds = {reds}"
    
    # fontScale 
    fontScale = 1
    
    # Blue color in BGR 
    color = blue 
    
    # Line thickness of 2 px 
    thickness = 2
    
    # Using cv2.putText() method 
    image = cv2.putText(frame, text, org, font,
                    fontScale, color, thickness, cv2.LINE_AA)


    ###############################
    #print(len(things))
    #print(things_counter)
    # cv2.imshow("RGB" ,frame)
    # cv2.waitKey(100)
    # cv2.imwrite(f"frame_{count}.jpg", frame)


    frames.append(frame)
    writer.write(frame)



writer.release()