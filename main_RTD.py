
import cv2
from ultralytics import YOLO

import random

from tracker import Tracker

capture = cv2.VideoCapture(0) # 0 => default camera
max_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
max_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)


model = YOLO("best.pt")


colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for c in range(10)]


tracker =Tracker()

detection_treshold = 0.30

while True:
    ret, frame = capture.read()

    if not ret:
        break 

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
            if score>detection_treshold:
                detections.append([x1,y1,x2,y2,score])

        tracker.update(frame, detections)

        for track in tracker.tracks:
            bbox = track.bbox
            x1,y1,x2,y2 = bbox
            track_id = track.track_id

            cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (colors[track_id % len(colors)]), 3)
            # cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
            #             colors[track_id % len(colors)], 2)
            cv2.putText(frame, f'Score: {score}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        colors[track_id % len(colors)], 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



capture.release()
cv2.destroyAllWindows()