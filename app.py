from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import cv2
from ultralytics import YOLO
import random
from tracker import Tracker
import subprocess
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

def process_video(video_path):
    model = YOLO("best.pt")
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    output_filename = f'output_{os.path.basename(video_path)}'

    captureOutput = cv2.VideoWriter(os.path.join('static', output_filename), cv2.VideoWriter_fourcc(*'MP4V'), 
                                    capture.get(cv2.CAP_PROP_FPS), 
                                    (frame.shape[1], frame.shape[0]))

    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for c in range(10)]
    tracker = Tracker()

    while ret:
        results = model(frame)

        for result in results:
            detections = []
            for r in result.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                class_id = int(class_id)
                detections.append([x1, y1, x2, y2, score])

            tracker.update(frame, detections)

            for track in tracker.tracks:
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 2)
                cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            colors[track_id % len(colors)], 2)
                cv2.putText(frame, f'Score: {score}', (int(x2), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        colors[track_id % len(colors)], 2)

        captureOutput.write(frame)
        ret, frame = capture.read()

    capture.release()
    captureOutput.release()
    cv2.destroyAllWindows()

    return output_filename

def process_image(image_path):
    model = YOLO("best.pt")
    results = model(image_path)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for c in range(10)]

    image = cv2.imread(image_path)
    for result in results:
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            detections.append([x1, y1, x2, y2, score])

        for detection in detections:
            x1, y1, x2, y2, score = detection
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f'Score: {score}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        colors[10 % len(colors)], 2)

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buffered = BytesIO()
    image_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/return')
def return_page():
    videos = os.listdir('static')
    videos_processed = [video for video in videos if video.startswith('output_')]
    processed_images = []

    images = os.listdir('static/images')
    for image in images:
        image_path = os.path.join('static/images', image)
        processed_image = process_image(image_path)
        processed_images.append(processed_image)

    return render_template('return.html', videos=videos_processed, images=processed_images)

@app.route('/open_with_media_player/<filename>')
def open_with_media_player(filename):
    video_path = os.path.join('static', filename)
    try:
        subprocess.Popen(['start', '', video_path], shell=True)
    except Exception as e:
        return f"Error: {e}"
    return "Opening video..."

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        if file_path.endswith(('.mp4', '.avi', '.mov')):
            output_filename = process_video(file_path)
        else:
            # For image files
            image_filename = os.path.basename(file_path)
            output_filename = f'output_{image_filename}'
            os.rename(file_path, os.path.join('static/images', image_filename))

        return redirect(url_for('return_page'))

if __name__ == '__main__':
    app.run(debug=True)
