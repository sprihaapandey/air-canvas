import cv2
import numpy as np
import mediapipe as mp

canvas = np.zeros((2000, 2000, 3), dtype="uint8")
brush_color = (255, 255, 255)
brush_size = 5
last_point = None
show_instructions = False
inst_button_width = 200
inst_button_height = 100
inst_box_width = 550
inst_box_height = 420

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def get_index_finger_position(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_finger_tip = hand_landmarks.landmark[8]
        
        height, width, _ = frame.shape
        x = int(index_finger_tip.x * width)
        y = int(index_finger_tip.y * height)
        
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return (x, y)
    return None

def event_handler(event, x, y, flags, param):
    global last_point, brush_color, brush_size, show_instructions
    if event == cv2.EVENT_LBUTTONUP:
        if 0 <= x <= inst_button_width and 0 <= y <= inst_button_height:
            show_instructions = not show_instructions
            return
        last_point = None

def handle_keys(key):
    global brush_color, brush_size
    if key == ord('r'):
        brush_color = (0, 0, 255)
    elif key == ord('g'):
        brush_color = (0, 255, 0)
    elif key == ord('b'):
        brush_color = (255, 0, 0)
    elif key == ord('w'):
        brush_color = (255, 255, 255)
    elif key == ord('e'):
        brush_color = (0, 0, 0)
    elif key == ord('+'):
        brush_size += 1
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)

def display_instructions(temp_canvas):
    cv2.rectangle(temp_canvas, (0, inst_button_height), (inst_box_width, inst_button_height + inst_box_height), (200, 200, 200), -1)
    instructions = [
        "Press 'r' for Red",
        "Press 'g' for Green",
        "Press 'b' for Blue",
        "Press 'w' for White",
        "Press 'e' for Eraser (Black)",
        "Press '+' to increase brush size",
        "Press '-' to decrease brush size",
        "Press 's' to save the painting",
        "Press 'q' to quit",
        "Use index finger to draw"
    ]
    y_offset = 140
    for instruction in instructions:
        cv2.putText(temp_canvas, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        y_offset += 40

# Main loop
cv2.namedWindow("Paint")
cv2.setMouseCallback("Paint", event_handler)
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break
        
    frame = cv2.flip(frame, 1)
    temp_canvas = canvas.copy()
    cv2.rectangle(temp_canvas, (0, 0), (inst_button_width, inst_button_height), (200, 200, 200), -1)
    cv2.putText(temp_canvas, "Instructions", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
    
    if show_instructions:
        display_instructions(temp_canvas)

    cv2.imshow("Camera Feed", frame)
    cv2.imshow("Paint", temp_canvas)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        handle_keys(key)
    if key == ord('s'):
        cv2.imwrite("my_painting.jpg", canvas)
        print("Painting saved as 'my_painting.jpg'")
        last_point = None
    elif key == ord('q'):
        break
    finger_pos = get_index_finger_position(frame)
    if finger_pos:
        x, y = finger_pos
        canvas_x = int(x * (canvas.shape[1] / frame.shape[1]))
        canvas_y = int(y * (canvas.shape[0] / frame.shape[0]))
        
        if last_point is not None:
            cv2.line(canvas, last_point, (canvas_x, canvas_y), brush_color, brush_size)
        last_point = (canvas_x, canvas_y)

    frame_resized = cv2.resize(frame, (canvas.shape[1], canvas.shape[0]))
    

cap.release()
cv2.destroyAllWindows()
hands.close()