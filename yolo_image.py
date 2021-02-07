# import necessary packages
import cv2
import numpy as np
import time
import json
import argparse
import os

'''
Arguments
'''
# json
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to the JSON configuration file")
args = vars(ap.parse_args())
# load the configuration
conf = json.load(open(args["conf"]))

'''
Image
'''
print("[YOLO] loading image from file ...")
# load input image
image = cv2.imread(conf["image_input"])
# get spatial dimensions from input image
h, w = image.shape[:2]

'''
Blob from image
'''
# construct a blob form the input image
blob = cv2.dnn.blobFromImage(image=image, scalefactor=1/255.0, size=(416,416), swapRB=True, crop=False)

'''
Load YOLO v3 network
'''
# load COCO class labels
labels_path = conf["yolo_coco"]
with open(labels_path) as f:
    # Getting labels reading every line and putting them into the list
    labels = [line.strip() for line in f]

# load YOLO trained model 
# load .weights and .cfg path
weights_path = conf["yolo_weights"]
config_path = conf["yolo_cfg"]
# load YOLO object detector
start_time = time.time()
print("[YOLO] loading YOLO model from disk...")
network = cv2.dnn.readNetFromDarknet(config_path, weights_path)
end_time = time.time()
elapsed_time = end_time - start_time
print("[YOLO] model loaded ... took {} seconds".format(elapsed_time))

# get list of all layers from YOLO network
layers_names_all = network.getLayerNames()
# get only output layer names that we need from YOLO
layers_names_output = [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# set minimum probability to filter weak detections
minimum_probability = conf["confidence"]

# set threshold when applying Non-Maxima Suppression
threshold = conf["threshold"]

# generate random colors for bounding boxes
np.random.seed(42)
colors = np.random.randint(low=0, high=255, size=(len(labels), 3), dtype="uint8")

'''
Forward pass
'''
# implement forward pass of the YOLO object detector with our blob and only through output layers
# calculate at the same time, needed time for forward pass
network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# show spent time for forward pass
print("[YOLO] objects detection took {:.6f} seconds".format(end - start))

'''
Get bounding boxes
'''
# initialize our lists of detected bounding boxes, confidences and class numbers
bounding_boxes = []
confidences = []
class_numbers = []

# loop over each of the layer outputs
for output in output_from_network:
    # loop over each of the detections
    for detected_object in output:
        # get 80 classes' probabilities for current detected object
        scores = detected_object[5:]
        # get index of the class with the maximum value of probability
        class_id = np.argmax(scores)
        # get value of probability for defined class
        confidence = scores[class_id]
        
        # filter weak predictions with minimum_probability
        if confidence > minimum_probability:
            # scale the bounding box coordinates to the initial image size
            # YOLO data format keeps coordinates for center of bounding box and its current width and height
            box_detected = detected_object[0:4] * np.array([w, h, w, h])
            
            # from YOLO data format, we can get top left corner coordinates that are x_min and y_min
            x_center, y_center, box_width, box_height = box_detected
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            
            # update our list of bounding box, confidences and class numbers
            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence))
            class_numbers.append(class_id)

'''
Non-Maximum Suppression
'''
# apply non-maximum suppression to suppress weak, overlapping bounding boxes
final_boxes = cv2.dnn.NMSBoxes(bounding_boxes, confidences, minimum_probability, threshold)

'''
Draw bounding boxes
'''
# defining counter for objects detected
counter = 1

# ensure at least one detection exists
if len(final_boxes) > 0:
    # loop over the indexes
    for i in final_boxes.flatten():
        # show labels of the detected objects
        print("[YOLO] object {0}: {1}".format(counter, labels[int(class_numbers[i])]))
        # Incrementing counter
        counter += 1

        # extract the bounding box coordinates
        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        
        # get the color
        box_color = colors[class_numbers[i]].tolist()
        
        # draw a bounding box rectangle
        cv2.rectangle(image, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      box_color, 2)
        
        # text with label and confidence on the image
        text_box = "{}: {:.1f}%".format(labels[int(class_numbers[i])], confidences[i] * 100)
        cv2.putText(image, text_box, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, box_color, 2)
        
'''
Display the image
'''
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
cv2.imshow("Detections", image)
# write new image with detections
cv2.imwrite(conf["image_output"], image)
# wait for any key being pressed
cv2.waitKey(0)
# destroy opened window
cv2.destroyWindow('Detections')        