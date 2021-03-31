import cv2
import numpy as np
import time 
import pyfakewebcam
import sys 
import torch
import torchvision
from Models import Segmenter

threshold = 0.5
colorDist = 20

frameModulo = 3

# Start a video cam session
video_session = cv2.VideoCapture(0)

#Grab the actual resolution of the webcam
fWidth  = int(video_session.get(cv2.CAP_PROP_FRAME_WIDTH))   
fHeight = int(video_session.get(cv2.CAP_PROP_FRAME_HEIGHT))  

#grab dummy webcam 
camera = pyfakewebcam.FakeWebcam('/dev/video2', fWidth, fHeight)

bgProvided = False 
if len(sys.argv) == 2:
    bgProvided = True
    path = sys.argv[1] 
    # Read the background image to memory
    bg_image = cv2.imread(path)
    bg_image = cv2.cvtColor(bg_image, cv2.COLOR_BGR2RGB)

    #This is gross, but I'm too lazy to do it smarter 

    #First crop the bg image to your target aspect ratio
    bgW, bgH, bgC = bg_image.shape 

    ratio = fWidth/fHeight 
    newbgH = 10
    newbgW = newbgH*(fWidth/fHeight)
    while newbgH < bgH and newbgW < bgW:
        newbgH += 1 
        newbgW = int(newbgH*ratio)

    bgWTrim = int((bgW - newbgW)/2)
    bgHTrim = int((bgH - newbgH)/2)

    if bgWTrim == 0:
        bgWTrim = 1 
    if bgHTrim == 0:
        bgHTrim = 1
    bg = bg_image[bgWTrim:-bgWTrim,bgHTrim:-bgHTrim]

    #Then resize it to our output size
    bg = cv2.resize(bg, (fWidth,fHeight))


model = Segmenter() 

start = time.time() 
count = 0 

estimatedBackground = None
mask = None
        
# Read frames from the video, make realtime predictions and display the same
while True:
    
    _, frame = video_session.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #TODO: centre crop the frame if desired 

    # Ensure there's something in the image (not completely blank)
    if np.any(frame):

        #start out with a simple average pixel for the background 
        if estimatedBackground is None:
            estimatedBackground = np.mean(np.mean(frame, axis=0), axis=0)
            estimatedBackground = np.full((fHeight, fWidth, 3), estimatedBackground)


        #To save time we only update our background estimate every few frames 
        #TODO: dynamically adjust the modulo here based on desired framerate 
        if count % frameModulo == 0:
            #get the predicted mask from our model 
            img = (frame/255.0).astype('float32')
            labels = model.pred(img)

            #adjust the mask to match the size of our video
            mask = cv2.resize(labels, (fWidth,fHeight))

            #update our estimate of the background image using the areas labeled as background 
            estimatedBackground[mask<threshold] = np.asarray((0.9*frame[mask<threshold]) + (0.1*estimatedBackground[mask<threshold]), dtype=np.uint8)

        #subtrack the background from the frame, and replace it with a green screen 
        dist = np.sum(np.abs(frame - estimatedBackground), axis=2)
        localmask = np.any([dist<colorDist, mask < threshold], axis=0) == False

        #blur the mask to remove jagged edges 
        localmask = np.asarray(localmask, dtype=np.float32)
        localmask = cv2.blur(localmask,(45,45))
        

        if bgProvided:
            #we want a smooth transition between humans and the background 
            #clip the mask so most values (appox below 0.45 and above 0.55) are pushed to 0 or 1 
            #but values inbetween are rescaled to the range (0,1)
            localmask = (localmask*9) - 4.0 
            localmask[localmask < 0] = 0 
            localmask[localmask > 1] = 1
            localmask = np.reshape(localmask, (fHeight,fWidth,1))
            localmask = np.concatenate([localmask,localmask,localmask], axis=2)

            #take the weighted average of the background and forground 
            #with the weights coming from our mask 

            frame = np.asarray((bg*(1-localmask)) + (frame*localmask), dtype=np.uint8) 

        else:
            
            frame[localmask < threshold] = [0,255,0]

        #display the frame 
        camera.schedule_frame(frame)

        
    else:
        break

    count += 1 

    end = time.time() 
    if end - start >= 1:
        print(count)
        count = 0 
        start = time.time() 

    
    
