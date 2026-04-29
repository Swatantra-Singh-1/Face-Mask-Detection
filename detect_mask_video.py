# importing necessary packages
import torch
import torch.nn as nn
from torchvision import transforms
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from PIL import Image

# get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Image preprocessing - must match training (112x112)
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Simple CNN model for mask detection (must match training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # 112 -> 56 -> 28 -> 14 -> 7 after 4 pools
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def detect_and_predict_mask(frame, faceNet, maskNet, device):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0), swapRB=False)

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.7:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# skip invalid boxes
			if startX >= endX or startY >= endY:
				continue

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 112x112, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = Image.fromarray(face)
			face = transform(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make predictions if at least one face was detected
	if len(faces) > 0:
		# stack faces into a batch
		faces = torch.stack(faces).to(device)
		
		# set model to eval mode and predict
		maskNet.eval()
		with torch.no_grad():
			outputs = maskNet(faces)
			probs = torch.softmax(outputs, dim=1).cpu().numpy()
		
		for prob in probs:
			preds.append(prob)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = os.path.join(SCRIPT_DIR, "face_detector", "deploy.prototxt")
weightsPath = os.path.join(SCRIPT_DIR, "face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading mask detector model...")
device = torch.device("cpu")

# Create model architecture matching training
maskNet = SimpleCNN()

# Load the saved weights
checkpoint = torch.load(os.path.join(SCRIPT_DIR, "mask_detector.pth"), map_location=device, weights_only=False)
maskNet.load_state_dict(checkpoint['model'])
maskNet = maskNet.to(device)
maskNet.eval()

print(f"[INFO] Model loaded. Using device: {device}")

# initialize the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet, device)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		with_mask_prob = pred[0]  # index 0 = with_mask
		without_mask_prob = pred[1]  # index 1 = without_mask

		# determine the class label and color we'll use to draw
		# the bounding box and text
		# Only show "Mask" if confidence is above 70%
		max_prob = max(with_mask_prob, without_mask_prob)
		if with_mask_prob > without_mask_prob and with_mask_prob > 0.7:
			label = "Mask"
			color = (0, 255, 0)
		elif without_mask_prob > with_mask_prob and without_mask_prob > 0.7:
			label = "No Mask"
			color = (0, 0, 255)
		else:
			label = "Unknown"
			color = (255, 255, 0)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max_prob * 100)

		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()