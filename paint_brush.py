import cv2
import numpy as np
import mediapipe as mp

canvas = np.zeros((720, 1280, 3), dtype="uint8")
brush_color = (255, 255, 255)
brush_size = 5
last_point = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

buttons = {
    'red': ((10, 10), (110, 110), (0, 0, 255)),
    'green': ((120, 10), (220, 110), (0, 255, 0)),
    'blue': ((230, 10), (330, 110), (255, 0, 0)),
    'white': ((340, 10), (440, 110), (255, 255, 255)),
    'eraser': ((450, 10), (550, 110), (0, 0, 0)),
    'small': ((570, 10), (670, 60), (200, 200, 200)),
    'medium': ((680, 10), (780, 60), (200, 200, 200)),
    'large': ((790, 10), (890, 60), (200, 200, 200))
}

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    temp_canvas = canvas.copy()

    for name, ((x1, y1), (x2, y2), color) in buttons.items():
        cv2.rectangle(temp_canvas, (x1, y1), (x2, y2), color, -1)
        text_color = (255, 255, 255) if name == 'eraser' else (0, 0, 0)
        cv2.putText(temp_canvas, name, (x1+5, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    finger_pos = None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[8]
        height, width, _ = frame.shape
        x = int(index_finger_tip.x * width)
        y = int(index_finger_tip.y * height)
        finger_pos = (x, y)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.circle(temp_canvas, (int(x * (1280/width)), int(y * (720/height))), 10, (0,255,255), -1)

        for name, ((x1, y1), (x2, y2), color) in buttons.items():
            if x1 < x < x2 and y1 < y < y2:
                if name == 'small':
                    brush_size = 5
                elif name == 'medium':
                    brush_size = 10
                elif name == 'large':
                    brush_size = 20
                else:
                    brush_color = color
                    last_point = None

        canvas_x = int(x * (canvas.shape[1] / frame.shape[1]))
        canvas_y = int(y * (canvas.shape[0] / frame.shape[0]))
        if last_point is not None:
            cv2.line(canvas, last_point, (canvas_x, canvas_y), brush_color, brush_size)
        last_point = (canvas_x, canvas_y)

    cv2.imshow("Paint", temp_canvas)
    cv2.imshow("Camera Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("my_painting.jpg", canvas)

cap.release()
cv2.destroyAllWindows()
hands.close()