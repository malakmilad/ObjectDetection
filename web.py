# object detection using web cam
##importing the library
import cv2
import numpy as np
import playsound
from gtts import gTTS


#the window
width = 320
height = 320

#how much you're sure about the detection
confidince_thrushold = 0.5

# Text to Speech Function
def speak(detected):
    for z, i in enumerate(detected):
        text = gTTS(text=i, lang='en')
        file_name = f'C:\\Users\\LOKA\\PycharmProjects\\ObjectDetection\\audio{z}.mp3'
        text.save(file_name)
        playsound.playsound(file_name)


#classes names
classes_path = "coco_classes.txt"
classNames = []
with open(classes_path, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#the weights of the model and the configration of the layers
model_configration = "yolov3-tiny_obj.cfg"
model_weghits = "yolov3.weights"

#passing the weights and cfgs to yolo network
net = cv2.dnn.readNetFromDarknet(model_configration, model_weghits)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#open the web cam
cap = cv2.VideoCapture(0)

global detected
detected = []

#drawing the bounding boxes around the detected objects
def find_objects(outputs, img):
    ht, wt, ct = image.shape
    bounding_box = []
    calssids = []
    confidince_value = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classid = np.argmax(scores)
            conf = scores[classid]
            if conf > confidince_thrushold:
                #print(classNames[classid])
                w, h = int(det[2]* wt) , int(det[3]* ht)
                x, y = int((det[0]*wt) - w/2), int((det[1]*ht) - h/2)
                bounding_box.append([x, y, w, h])
                confidince_value.append(float(conf))
                calssids.append(classid)

    #print(len(bounding_box))
    incidies = cv2.dnn.NMSBoxes(bounding_box, confidince_value, confidince_thrushold, nms_threshold=0.3)
    for i in incidies:
        # i = i[0]
        box = bounding_box[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (w+x, h+y),(255, 0, 255), 2)
        cv2.putText(img, f'{classNames[calssids[i]].upper()}, {int(confidince_value[i]*100)}%',
                    (x, y-10), cv2.FONT_ITALIC, 0.6, (255, 0, 0), 2)
        detected.append(classNames[calssids[i]])
        print(detected)
        #print(classNames[calssids[i]])


# for videos or live streaming
print(detected)
while True:
    success, image = cap.read()
    blob = cv2.dnn.blobFromImage(image, 1/255, (width, height), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)
    find_objects(outputs, image)
    cv2.imshow("objectdetection", image)
    speak(detected)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


