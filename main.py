import os
import cv2
from ultralytics import YOLO
import random
from tracker import Tracker

videoPath = os.path.join('.','data','maj.mp4') 
videoOutPath = os.path.join('.',"output.mp4")
capture = cv2.VideoCapture(videoPath)
ret, frame = capture.read()
captureOutput = cv2.VideoWriter(videoOutPath, cv2.VideoWriter_fourcc(*'MP4V'), 
                                capture.get(cv2.CAP_PROP_FPS), 
                                (frame.shape[1],frame.shape[0]))

model = YOLO("best.pt")
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for c in range(10)]
tracker =Tracker()
while ret:

    results = model(frame)

    for result in results:
        detections =[]
        for r in result.boxes.data.tolist():
            x1,y1,x2,y2,score,class_id =r
            x1=int(x1)
            x2=int(x2)
            y1=int(y1)
            y2=int(y2)
            class_id=int(class_id)
            detections.append([x1,y1,x2,y2,score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1,y1,x2,y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (colors[track_id % len(colors)]), 2)
            cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       colors[track_id % len(colors)], 2)
    # cv2.imshow('frame',frame)
    # cv2.waitKey(25)
    captureOutput.write(frame)
    ret, frame = capture.read()

capture.release()
captureOutput.release()
cv2.destroyAllWindows()