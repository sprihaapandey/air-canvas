from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
import os

app = Flask(__name__)

class PaintApp:
    def __init__(self):
        self.canvas = np.zeros((480, 1000, 3), dtype="uint8")  # Wider canvas
        self.brush_color = (255, 255, 255)
        self.brush_size = 5
        self.last_point = None
        self.finger_pos = None
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1, 
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        self.current_frame = None
        
        self.camera_thread = threading.Thread(target=self.camera_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
    
    def camera_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue
                
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                index_finger_tip = hand_landmarks.landmark[8]
                height, width, _ = frame.shape
                x = int(index_finger_tip.x * width)
                y = int(index_finger_tip.y * height)
                
                self.finger_pos = (x, y)
                
                canvas_x = int(x * (self.canvas.shape[1] / width))
                canvas_y = int(y * (self.canvas.shape[0] / height))
                
                self.check_button_interaction(canvas_x, canvas_y)
                
                if not self.is_over_button(canvas_x, canvas_y):
                    if self.last_point is not None:
                        cv2.line(self.canvas, self.last_point, (canvas_x, canvas_y), 
                                self.brush_color, self.brush_size)
                    self.last_point = (canvas_x, canvas_y)
                else:
                    self.last_point = None
                
                self.mp_draw.draw_landmarks(
                    self.current_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
            else:
                self.finger_pos = None
                self.last_point = None
            
            time.sleep(0.03)
    
    def get_canvas_with_ui(self):
        temp_canvas = self.canvas.copy()

        buttons = {
            'red':    ((10, 10),   (110, 60),   (0, 0, 255)),
            'green':  ((120, 10),  (220, 60),   (0, 255, 0)),
            'blue':   ((230, 10),  (330, 60),   (255, 0, 0)),
            'white':  ((340, 10),  (440, 60),   (255, 255, 255)),
            'eraser': ((450, 10),  (550, 60),   (0, 0, 0)),

            'small':  ((560, 10),  (660, 60),   (200, 200, 200)),
            'medium': ((670, 10),  (770, 60),   (200, 200, 200)),
            'large':  ((780, 10),  (880, 60),   (200, 200, 200))
        }

        for name, ((x1, y1), (x2, y2), color) in buttons.items():
            if ((name == 'red' and self.brush_color == (0, 0, 255)) or
                (name == 'green' and self.brush_color == (0, 255, 0)) or
                (name == 'blue' and self.brush_color == (255, 0, 0)) or
                (name == 'white' and self.brush_color == (255, 255, 255)) or
                (name == 'eraser' and self.brush_color == (0, 0, 0)) or
                (name == 'small' and self.brush_size == 5) or
                (name == 'medium' and self.brush_size == 10) or
                (name == 'large' and self.brush_size == 20)):
                cv2.rectangle(temp_canvas, (x1-3, y1-3), (x2+3, y2+3), (0, 255, 255), 2)

            cv2.rectangle(temp_canvas, (x1, y1), (x2, y2), color, -1)
            text_color = (255, 255, 255) if name == 'eraser' else (0, 0, 0)
            font_scale = 0.6
            text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_x = x1 + (x2 - x1 - text_size[0]) // 2
            text_y = y2 - 15
            cv2.putText(temp_canvas, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1)

        # Finger position pointer
        if self.finger_pos:
            x, y = self.finger_pos
            canvas_x = int(x * (temp_canvas.shape[1] / 640))
            canvas_y = int(y * (temp_canvas.shape[0] / 480))
            cv2.circle(temp_canvas, (canvas_x, canvas_y), 8, (0, 255, 255), -1)

        # Current brush info
        color_name = self.get_color_name()
        size_name = self.get_size_name()
        info_text = f"{color_name} | Size: {size_name}"
        cv2.putText(temp_canvas, info_text, (10, temp_canvas.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return temp_canvas
    
    def get_color_name(self):
        color_map = {
            (0, 0, 255): 'Red',
            (0, 255, 0): 'Green',
            (255, 0, 0): 'Blue',
            (255, 255, 255): 'White',
            (0, 0, 0): 'Eraser'
        }
        return color_map.get(self.brush_color, 'Unknown')
    
    def get_size_name(self):
        size_map = {5: 'Small', 10: 'Medium', 20: 'Large'}
        return size_map.get(self.brush_size, 'Unknown')
    
    def set_brush_color(self, color):
        color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'white': (255, 255, 255),
            'eraser': (0, 0, 0)
        }
        self.brush_color = color_map.get(color, (255, 255, 255))
        self.last_point = None
    
    def set_brush_size(self, size):
        size_map = {'small': 5, 'medium': 10, 'large': 20}
        self.brush_size = size_map.get(size, 5)
    
    def check_button_interaction(self, x, y):
        buttons = {
            'red': ((10, 10), (110, 60)),
            'green': ((120, 10), (220, 60)),
            'blue': ((230, 10), (330, 60)),
            'white': ((340, 10), (440, 60)),
            'eraser': ((450, 10), (550, 60)),
            'small': ((560, 10), (660, 60)),
            'medium': ((670, 10), (770, 60)),
            'large': ((780, 10), (880, 60))
        }
        
        for name, ((x1, y1), (x2, y2)) in buttons.items():
            if x1 < x < x2 and y1 < y < y2:
                if name in ['small', 'medium', 'large']:
                    self.set_brush_size(name)
                else:
                    self.set_brush_color(name)
                self.last_point = None
                break
    
    def is_over_button(self, x, y):
        buttons = [
            ((10, 10), (110, 60)),
            ((120, 10), (220, 60)),
            ((230, 10), (330, 60)),
            ((340, 10), (440, 60)),
            ((450, 10), (550, 60)),
            ((560, 10), (660, 60)),
            ((670, 10), (770, 60)),
            ((780, 10), (880, 60))
        ]
        
        for (x1, y1), (x2, y2) in buttons:
            if x1 < x < x2 and y1 < y < y2:
                return True
        return False
    
    def clear_canvas(self):
        self.canvas = np.zeros((480, 1000, 3), dtype="uint8")
    
    def save_painting(self, filename="painting.jpg"):
        cv2.imwrite(filename, self.canvas)
        return filename
    
    def __del__(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()
        if hasattr(self, 'hands'):
            self.hands.close()

paint_app = PaintApp()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if paint_app.current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', paint_app.current_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/canvas_feed')
def canvas_feed():
    def generate():
        while True:
            canvas_with_ui = paint_app.get_canvas_with_ui()
            ret, buffer = cv2.imencode('.jpg', canvas_with_ui)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_color', methods=['POST'])
def set_color():
    color = request.json.get('color')
    paint_app.set_brush_color(color)
    return jsonify({'status': 'success', 'color': color})

@app.route('/set_size', methods=['POST'])
def set_size():
    size = request.json.get('size')
    paint_app.set_brush_size(size)
    return jsonify({'status': 'success', 'size': size})

@app.route('/clear', methods=['POST'])
def clear():
    paint_app.clear_canvas()
    return jsonify({'status': 'success'})

@app.route('/save', methods=['POST'])
def save():
    filename = paint_app.save_painting()
    return jsonify({'status': 'success', 'filename': filename})

@app.route('/download')
def download():
    return send_file('painting.jpg', as_attachment=True, download_name='my_painting.jpg')

# HTML Template
html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Python Hand Tracking Paint</title>
    <style>
        body {
            margin: 0;
            padding: 20px;
            background: #2c2c2c;
            font-family: Arial, sans-serif;
            color: white;
        }

        .container {
            display: flex;
            gap: 20px;
            justify-content: center;
            align-items: flex-start;
            flex-wrap: wrap;
        }

        .video-container {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .video-feed {
            border: 3px solid #fff;
            border-radius: 10px;
            width: 1000px;
            height: 480px;
            object-fit: cover;
        }

        .controls {
            display: flex;
            flex-direction: column;
            gap: 15px;
            padding: 20px;
            background: #3c3c3c;
            border-radius: 10px;
            min-width: 200px;
            max-width: 250px;
        }

        .action-btn {
            padding: 15px 20px;
            font-size: 16px;
            width: 100%;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }

        .action-btn:hover {
            transform: scale(1.05);
        }

        .save-btn {
            background: #4caf50;
            color: white;
        }

        .clear-btn {
            background: #f44336;
            color: white;
        }

        h1 {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }

        h3 {
            margin: 10px 0 5px 0;
            color: #ddd;
            text-align: center;
        }

        .status {
            background: #444;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            margin-bottom: 20px;
        }

        .instructions {
            background: #444;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 10px;
            font-size: 14px;
            line-height: 1.4;
        }
    </style>
</head>
<body>
    <h1>🎨 Python Hand Tracking Paint</h1>

    <div class="status" id="status">
        Show your index finger to the camera to start drawing!
    </div>

    <div class="container">
        <div class="video-container">
            <h3>🎨 Drawing Canvas</h3>
            <img src="/canvas_feed" alt="Canvas" class="video-feed" />
        </div>

        <div class="controls">
            <div class="instructions">
                <strong>How to use:</strong><br />
                • Point your finger at the colored buttons in the canvas to change colors<br />
                • Use the size buttons (small/medium/large) to change brush size<br />
                • Move your finger anywhere else to draw<br />
                • Current settings are highlighted with cyan borders
            </div>

            <h3>Actions</h3>
            <button class="action-btn save-btn" onclick="savePainting()">💾 Save Painting</button>
            <button class="action-btn clear-btn" onclick="clearCanvas()">🗑️ Clear Canvas</button>
        </div>
    </div>

    <script>
        function clearCanvas() {
            fetch('/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            document.getElementById('status').textContent = 'Canvas cleared!';
        }

        function savePainting() {
            fetch('/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            }).then(() => {
                window.location.href = '/download';
                document.getElementById('status').textContent = 'Painting saved and downloaded!';
            });
        }

        document.addEventListener('keydown', (e) => {
            switch (e.key.toLowerCase()) {
                case 's':
                    savePainting();
                    break;
                case 'c':
                    clearCanvas();
                    break;
            }
        });
    </script>
</body>
</html>
'''

if not os.path.exists('templates'):
    os.makedirs('templates')

with open('templates/index.html', 'w') as f:
    f.write(html_template)

if __name__ == '__main__':
    print("Starting Python Hand Tracking Paint Web App...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        paint_app.running = False
