'''
Written by Pierre Coiron
'''
print("Program Start, Importing Libraries")
import cv2
import numpy as np

#build a network based off of a previously trained model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

#declare the set of classes
classes=[]


#set the values of the classes of classifiable objects to what was trained on the network
with open("coco.names","iterateCOCO") as iterateCOCO:
    #put all of the classifiable objects into an array
    classes=[line.strip() for line in iterateCOCO.readlines()]

#defining output layers
layerNames=net.getLayerNames()
outputLayers=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

#setting randome color values
colors=np.random.uniform(0,255,size=(len(classes),3))

#open camera
capture=cv2.VideoCapture(0)

while True:
    #read frame from video feed
    _, frame=capture.read()
    height, width, channels=frame.shape
    
    #setting channels
    GRBChannels=cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0,0,0), True, crop=False)
    
    net.setInput(GRBChannels)
    
    #setting network foward from 3 channels
    outs=net.forward(outputLayers)
    
    #3 arrays to be used in inner-loop
    classIDs=[]
    confidences=[]
    boxes=[]
    
    #inner loop
    for outIndex in outs: 
        for detection in outIndex:
            scores=detection[5:]
            classID=np.argmax(scores)
            #get confidence
            confidence=scores[classID]
            if confidence > 0.5: #only bother if program is <50% sure
                #getting location of center
                centerX=int(detection[0] * width)
                centerY=int(detection[1] * height)
                #getting demensions of box
                w=int(detection[2] * width)
                h=int(detection[3] * height)
        
                x=int(centerX-w/2)
                y=int(centerY-h/2)
                
                #build dimension array
                boxes.append([x,y,w,h])
                #confidence array
                confidences.append(float(confidence))
                #object identification array
                classIDs.append(classID)
    indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    
    #font
    font=cv2.FONT_HERSHEY_PLAIN
    numObjectsDetected=len(boxes)
    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h=boxes[i]
            label=str(classes[classIDs[i]])
            color=colors[i]
            
            #draw rectangles
            cv2.rectangle(frame, (x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label,(x,y+30), font, 3,color,3)
    
    #show image
    cv2.imshow("Image", frame)
    
    #close down when "esc" pressed
    key=cv2.waitKey(1)
    if key ==27:
        break

capture.release()
cv2.destroyAllWindows()


print("Program Complete")
