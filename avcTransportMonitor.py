import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from imutils.video import VideoStream
from datetime import datetime
from imutils.video import FPS
import argparse
import imutils
import time
import cv2
import os
import anothernumplatereg

# predefined values
CONST_CONFIDENCE=0.2
PATH = r"D:\LAB\pyhton--ml\tkinter video"
FORMAT='.jpg'
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#loading models
#print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

#Set up GUI
window = tk.Tk()  #Makes main window
window.wm_title("AVC College Transport")
window.config(background="#FFFFFF")

#print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0) 
fps = FPS().start()

#Graphics window
cameraLabel=tk.Label(window, text="Camera Stream")
cameraLabel.grid(row=0, column=0, padx=10, pady=2)
imageFrame = tk.Frame(window, width=800, height=700)
imageFrame.grid(row=1, column=0, padx=10, pady=2)

#Capture video frames
lmain = tk.Label(imageFrame)
lmain.grid(row=0, column=0)
cap = cv2.VideoCapture(0)

def show_frame():
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
    0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > CONST_CONFIDENCE:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            # label = "{} : {:.2f}%".format(CLASSES[idx],
            # confidence * 100)

            # cv2.rectangle(frame, (startX, startY), (endX, endY),
            # COLORS[idx], 2)
            # y = startY - 15 if startY - 15 > 15 else startY + 15
            # cv2.putText(frame, label, (startX, y),
            # cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            if(CLASSES[idx]=="bus" or CLASSES[idx]=="car"):
                top = tk.Toplevel()
                top.title('Detection')
                now=datetime.now()
                timestamp=str(now)
                timestamp=timestamp.replace(":","_")
                fullpath=os.path.join(PATH , timestamp+FORMAT)
                print(fullpath)
                time.sleep(2)
                testFullPath="D:\\LAB\\pyhton--ml\\car.jpg"
                val=cv2.imwrite(fullpath,frame)
                if val:
                    print("number"+anothernumplatereg.detectNumberPlateFromImagePath(testFullPath))
                tk.Message(top, text="A car/Bus "+str(val)+" has been detected "+timestamp, padx=20, pady=20).pack()
                top.after(3000, top.destroy)
                # tk.messagebox.showinfo("Title", "a Tk MessageBox")
                

    # _, frame = cap.read()
    fps.update()
    # frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame) 

#Slider window (slider controls stage position)
sliderFrame = tk.Frame(window, width=600, height=100)
sliderFrame.grid(row = 6, column=0, padx=10, pady=2)

show_frame()  #Display 2
window.mainloop()  #Starts GUI