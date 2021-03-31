# A Simple Virtual Green Screen Application

A simple virtual green screen application which relies on pre-trained image segmentation models combined with background subtractions. The basic idea is to use the image segmentation model to identify the humans, and use anything not labeled as a human to build up an estimate of the background. The background estimate is constantly updated based on recent frames, so it can adapt to changes in lighting, some movement, etc. Our final segmentation is a combination of the models prediction, and background subtraction. This helps adjust for any noise/jitter in the models prediction, and lets us avoid using the model on every frame, increasing framerate. 

This repo relies on v4l2loopback to create a virtual webcam. You can run modprobe v4l2loopback devices=1 exclusive_caps=1 as root to initialize a virtual webcam and modprobe -r v4l2loopback to unload the module if needed. Make sure pyfakewebcam is pointing at the correct device(s) in /dev. The exclusive_caps flag is needed for the virtual camera to be recognized by chrome. Also, don't forget to update

You can play around with the frameModulo parameter if you want to increase framerate, or attempt to run on CPU. At default we'll use the pytorch model once every 3 frames to segment the person and update our estimate of what the background looks like. Increasing frameModulo means you use the model less often, increasing framerate but decreasing performance. 

You can run python3 VirtualGreenScreen.py to capture your webcam and insert a virutal green screen. Alternatively you can provide a background image as a single command line argument. 

This script was inspired by this repo https://github.com/ElisonSherton/instanceSegmentation/, but improves on framerate and the tightness of edges around the subject. 