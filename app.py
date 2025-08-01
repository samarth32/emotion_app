import cv2
import numpy as np
import os
from flask import Flask, render_template, Response, request, jsonify
import face_recognition
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import sqlite3
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from PIL import Image

app = Flask(__name__)


# Initialize database
def init_db():
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS faces
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  image_path TEXT NOT NULL,
                  date_added TEXT NOT NULL)''')
    conn.commit()
    conn.close()


init_db()

latest_results = []
detection_lock = threading.Lock()
is_detecting = False

# Load Emotion Model
emotion_model_path = "fer2013_mini_XCEPTION.102-0.66.hdf5"
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["Angry", "Disgust", "Scared", "Happy", "Sad", "Surprised", "Neutral"]

# Load Animated Emoji GIFs
emoji_animations = {}
emoji_folder = "C:\\Users\\SAMARTH\\Desktop\\Odoo-Community Days\\Emojis"
for emotion in EMOTIONS:
    gif_path = os.path.join(emoji_folder, f"{emotion}.gif")
    if os.path.exists(gif_path):
        try:
            gif = Image.open(gif_path)
            frames = []
            while True:
                frame = gif.copy().convert("RGBA")
                frames.append(np.array(frame))
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        emoji_animations[emotion] = frames
    else:
        print(f"⚠️ GIF emoji not found for: {emotion}")

# Track current frame index for each emotion
emoji_frame_index = {emotion: 0 for emotion in EMOTIONS}

# Globals
last_captured_frame = None
known_face_encodings = []
known_face_names = []
last_face_update = 0
executor = ThreadPoolExecutor(max_workers=2)
processing_frame = False


def detect_emotion(face_img):
    try:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        gray = gray.astype("float") / 255.0
        gray = img_to_array(gray)
        gray = np.expand_dims(gray, axis=0)
        preds = emotion_classifier.predict(gray, verbose=0)[0]
        return EMOTIONS[np.argmax(preds)]
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "Neutral"


def load_known_faces():
    global known_face_encodings, known_face_names, last_face_update

    if time.time() - last_face_update < 10:
        return

    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT name, image_path FROM faces")
        known_faces = c.fetchall()
        conn.close()

        new_encodings = []
        new_names = []

        for name, image_path in known_faces:
            if not os.path.exists(image_path):
                continue

            try:
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    new_encodings.append(encodings[0])
                    new_names.append(name)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        known_face_encodings = new_encodings
        known_face_names = new_names
        last_face_update = time.time()

    except Exception as e:
        print(f"Error loading known faces: {e}")


def recognize_face(face_img):
    try:
        load_known_faces()

        if not known_face_encodings:
            return "Unknown Face"

        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face)

        if not face_encodings:
            return "Unknown Face"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[0])
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        if best_distance <= 0.45:
            return known_face_names[best_match_index]
        else:
            return "Unknown Face"

    except Exception as e:
        print(f"Face recognition error: {e}")
        return "Unknown Face"


def background_detect_and_recognize(frame):
    global latest_results, is_detecting

    try:
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        results = []
        for (top, right, bottom, left) in face_locations:
            top, right, bottom, left = [x * 2 for x in (top, right, bottom, left)]
            face_img = frame[top:bottom, left:right]

            emotion = detect_emotion(face_img)
            name = recognize_face(face_img)

            results.append((top, right, bottom, left, name, emotion))

        with detection_lock:
            latest_results = results

    except Exception as e:
        print(f"[Detection Error]: {e}")
    finally:
        is_detecting = False


def apply_gamma_correction(image, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def auto_white_balance(image):
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def gen_frames():
    global last_captured_frame, latest_results, is_detecting

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not accessible.")
        return

    # Set HD resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_count = 0
    detection_interval = 5
    current_emotion = None
    face_detected = False

    # Frame dimensions
    camera_width = 640
    camera_height = 480
    emoji_width = 640
    emoji_height = 480

    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to read frame.")
            continue

        # Apply enhancements
        frame = auto_white_balance(frame)
        frame = apply_gamma_correction(frame, gamma=1.3)
        frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

        last_captured_frame = frame.copy()
        frame_count += 1

        if frame_count % detection_interval == 0 and not is_detecting:
            is_detecting = True
            threading.Thread(target=background_detect_and_recognize, args=(frame.copy(),), daemon=True).start()

        emoji_display = np.full((emoji_height, emoji_width, 3), 255, dtype=np.uint8)
        face_detected = False

        with detection_lock:
            if latest_results:
                face_detected = True
                current_emotion = latest_results[0][5]

                for (top, right, bottom, left, name, emotion) in latest_results:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    name_color = (0, 0, 255) if name == "Unknown Face" else (0, 255, 0)
                    cv2.putText(frame, name, (left, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, name_color, 2)

            if face_detected and current_emotion and current_emotion in emoji_animations:
                frames = emoji_animations[current_emotion]
                if frames:
                    frame_idx = emoji_frame_index[current_emotion]
                    emoji = frames[frame_idx % len(frames)]
                    emoji_frame_index[current_emotion] = (frame_idx + 1) % len(frames)

                    emoji_resized = cv2.resize(emoji, (200, 200), interpolation=cv2.INTER_LANCZOS4)
                    alpha_channel = emoji_resized[:, :, 3] / 255.0
                    inverse_alpha = 1.0 - alpha_channel
                    x_offset = (emoji_width - 200) // 2
                    y_offset = (emoji_height - 200) // 2

                    for c in range(3):
                        emoji_display[y_offset:y_offset + 200, x_offset:x_offset + 200, c] = (
                                alpha_channel * emoji_resized[:, :, c] +
                                inverse_alpha * emoji_display[y_offset:y_offset + 200, x_offset:x_offset + 200, c]
                        )

                    cv2.putText(emoji_display, f"Emotion: {current_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 0), 2)
            else:
                cv2.putText(emoji_display, "No face detected", (emoji_width // 4, emoji_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                current_emotion = None

        # Resize for display
        camera_frame = cv2.resize(frame, (camera_width, camera_height), interpolation=cv2.INTER_AREA)
        combined_frame = np.hstack((camera_frame, emoji_display))

        ret, buffer = cv2.imencode('.jpg', combined_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_face', methods=['POST'])
def capture_face():
    global last_captured_frame, last_face_update

    try:
        data = request.get_json()
        face_name = data['name'].strip()

        if not face_name:
            return jsonify({'success': False, 'error': 'Name cannot be empty'})

        if last_captured_frame is None:
            return jsonify({'success': False, 'error': 'No frame captured'})

        faces_dir = os.path.join('static', 'faces')
        os.makedirs(faces_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{face_name}_{timestamp}.jpg"
        filepath = os.path.join(faces_dir, filename)

        cv2.imwrite(filepath, last_captured_frame)

        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("INSERT INTO faces (name, image_path, date_added) VALUES (?, ?, ?)",
                  (face_name, filepath, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

        last_face_update = 0

        return jsonify({'success': True, 'filepath': filepath})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_faces')
def get_faces():
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute("SELECT id, name, image_path, date_added FROM faces ORDER BY date_added DESC")
        faces = [{'id': row[0], 'name': row[1], 'image_path': row[2], 'date_added': row[3]}
                 for row in c.fetchall()]
        conn.close()
        return jsonify({'success': True, 'faces': faces, 'count': len(faces)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/delete_face/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()

        # First get the image path so we can delete the file
        c.execute("SELECT image_path FROM faces WHERE id = ?", (face_id,))
        result = c.fetchone()

        if not result:
            return jsonify({'success': False, 'error': 'Face not found'})

        image_path = result[0]

        # Delete from database
        c.execute("DELETE FROM faces WHERE id = ?", (face_id,))
        conn.commit()
        conn.close()

        # Delete the image file
        if os.path.exists(image_path):
            os.remove(image_path)

        # Reset face encodings cache
        global last_face_update
        last_face_update = 0

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)