import cv2
import numpy as np
import urllib.request
import os
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
 
MODEL_PATH = 'face_landmarker.task'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
 
if not os.path.exists(MODEL_PATH):
    print("Скачиваю модель face_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
 
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
)
detector = vision.FaceLandmarker.create_from_options(options)
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = "C:/Users/RedFox3k/hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # большой палец
    (0, 5), (5, 6), (6, 7), (7, 8),          # указательный
    (5, 9), (9, 10), (10, 11), (11, 12),     # средний
    (9, 13), (13, 14), (14, 15), (15, 16),   # безымянный
    (13, 17), (17, 18), (18, 19), (19, 20),  # мизинец
    (0, 17),                                  # ладонь
]

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HandLandmarker.create_from_options(options)

TIP_IDS = [4, 8, 12, 16, 20]

def distance(p_a,p_b):
    return math.hypot(p_a.x - p_b.x, p_a.y - p_b.y)

def count_fingers(landmarks, handedness_label):
    fingers = []
    thumb_tip = landmarks[4]
    index_mcp = landmarks[17]
    t_i_d = distance(thumb_tip, index_mcp)
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    hand_size = distance(wrist, middle_mcp)
    THUMB_EXTENDED_RATIO = 0.8
    fingers.append(1 if (t_i_d / hand_size)>THUMB_EXTENDED_RATIO else 0)
    
    for i in range(1,5):
        tip = TIP_IDS[i]
        pip_joint = tip - 2
        fingers.append(1 if landmarks[tip].y < landmarks[pip_joint].y else 0)
    return fingers 
 
 
def find_reaction_image(folder, basename):
    if not os.path.isdir(folder):
        return None
    for filename in os.listdir(folder):
        name_without_ext, ext = os.path.splitext(filename)
        if name_without_ext.lower() == basename.lower() and ext.lower() in (
            ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp",
        ):
            return os.path.join(folder, filename)
    return None

def get_blandshapes_dict(detection_result, face_index=0):
    if not detection_result.face_blendshapes:
        return{}
    categoties = detection_result.face_blendshapes[face_index]
    return {c.category_name: c.score for c in categoties}
 
 

WINK_THRESHOLD = 0.3
WINK_OPEN_THRESHOLD = 0.3
JAW_OPEN_THRESHOLD = 0.35
SMILE_THRESHOLD = 0.7
BROW_UP_THRESHOLD = 0.4
PUCKER_THRESHOLD = 0.4
FROWN_THRESHOLD = 0.4

 
def get_blendshapes_dict(detection_result, face_index=0):
    if not detection_result.face_blendshapes:
        return {}
    categories = detection_result.face_blendshapes[face_index]
    return {c.category_name: c.score for c in categories}
 
 

def check_wink_tongue(ctx):
    b = ctx['b']
    left = b.get('eyeBlinkLeft', 0.0)
    right = b.get('eyeBlinkRight', 0.0)
    winking = (left > WINK_THRESHOLD and right < WINK_OPEN_THRESHOLD) or \
              (right > WINK_THRESHOLD and left < WINK_OPEN_THRESHOLD)
    
    mouth_open = ctx["jaw_open"] > JAW_OPEN_THRESHOLD
    return winking and  mouth_open

def check_smile(ctx):
    b = ctx['b']
    smile = b.get('mouthSmileLeft', 0.0) + b.get('mouthSmileRight', 0.0)
    return smile / 2 > SMILE_THRESHOLD

        
EXPRESSIONS = []

for expr in EXPRESSIONS:
    path = find_reaction_image(SCRIPT_DIR, expr['image_basename'])
    if path is None:
        expr['image'] = None
        print(f"⚠ [{expr['name']}] не найден файл \"{expr['image_basename']}.*\" в папке {SCRIPT_DIR}")
    else:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠ [{expr['name']}] файл найден ({path}), но не читается как изображение.")
        expr["image"] = img
        
FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]
 
LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]
 
RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
]
 
LEFT_EYEBROW = [
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
]
 
RIGHT_EYEBROW = [
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
]
 
LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
]
 
ALL_CONNECTIONS = FACE_OVAL + LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS
 
 
def draw_face_mesh(frame, face_landmarks):
    h, w, _ = frame.shape
 
    points = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
    )
    for start_idx, end_idx in ALL_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 1, cv2.LINE_AA)
 
    for (x, y) in points:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1, cv2.LINE_AA)
 
    return frame

def draw_hand(frame, landmarks_px):
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, landmarks_px[start_idx], landmarks_px[end_idx],
(0, 255, 0), 2)
    for x,y in landmarks_px:
        cv2.circle(frame,(x,y),4,(0,255,0), -1)
 
 
cap = cv2.VideoCapture(0)
frame_timestamps_ms = 0
reaction_window_open = False
 
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
 
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
 
    frame_timestamps_ms += 33
    detection_result = detector.detect_for_video(mp_image, frame_timestamps_ms)
    result = landmarker.detect_for_video(mp_image, frame_timestamps_ms)
    triggered_expr = None
 
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            frame = draw_face_mesh(frame, face_landmarks)
 
        blendshapes = get_blendshapes_dict(detection_result)
        face_landmarks = detection_result.face_landmarks[0]
        ctx = {
            "b": blendshapes,
            "jaw_open": blendshapes.get("jawOpen", 0.0),
        }
 
       
        for expr in EXPRESSIONS:
            if expr["check"](ctx):
                triggered_expr = expr
                break
 
        status_text = triggered_expr["name"] if triggered_expr else "..."
        cv2.putText(
            frame,
            f"expression: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
 
        debug_lines = [
            f"blink L/R={ctx['b'].get('eyeBlinkLeft',0):.2f}/{ctx['b'].get('eyeBlinkRight',0):.2f} "
            f"jaw={ctx['jaw_open']:.2f}",
            f"smile L/R={ctx['b'].get('mouthSmileLeft',0):.2f}/{ctx['b'].get('mouthSmileRight',0):.2f} "
            f"browUp={ctx['b'].get('browInnerUp',0):.2f}",
            f"frown L/R={ctx['b'].get('browDownLeft',0):.2f}/{ctx['b'].get('browDownRight',0):.2f} "
            f"pucker={ctx['b'].get('mouthPucker',0):.2f}",
        ]
        for i, line in enumerate(debug_lines):
            cv2.putText(
                frame, line, (10, 60 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
 
    if result.hand_landmarks:
            for hand_index,(hand_landmarks, handedness) in  enumerate(zip(result.hand_landmarks, result.handedness)):
                landmarks_px = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]
                draw_hand(frame, landmarks_px) 
                label = handedness[0].category_name
        
    if triggered_expr is not None and triggered_expr.get('image') is not None:
        cv2.imshow('Reaction', triggered_expr['image'])
        reaction_window_open = True
    elif reaction_window_open:
        cv2.destroyWindow('Reaction')
        reaction_window_open = False
        
    cv2.imshow('Face Skeleton', frame)
    
 
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
detector.close()
 
