import cv2
import numpy as np
import sys

COLOR_RANGES = {
    'red1':  ((0, 120, 50),  (8, 255, 255)),
    'red2':  ((170, 120, 50), (180, 255, 255)),
    'yellow': ((15, 100, 120), (35, 255, 255)),
    'green': ((36, 80, 80), (89, 255, 255)),
}

VIS_COLORS = {
    'red': (0,0,255),
    'yellow': (0,255,255),
    'green': (0,255,0),
    'unknown': (200,200,200)
}

def get_color_mask(hsv, color_name):
    if color_name == 'red':
        lower1, upper1 = COLOR_RANGES['red1']
        lower2, upper2 = COLOR_RANGES['red2']
        m1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
        m2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
        return cv2.bitwise_or(m1, m2)
    else:
        low, up = COLOR_RANGES[color_name]
        return cv2.inRange(hsv, np.array(low), np.array(up))

def find_valid_contours(mask, min_area=200):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x,y,w,h = cv2.boundingRect(c)
        valid.append({'area': area, 'bbox': (x,y,w,h)})
    return valid

def classify_state(detections):
    totals = {c: sum(d['area'] for d in dets) for c,dets in detections.items()}
    if not any(totals.values()):
        return 'UNKNOWN'
    selected = max(totals.items(), key=lambda x: x[1])
    return selected[0].upper()

def main():
    if len(sys.argv) < 2:
        print("Usage: python traffic_light_detector.py <video_file>")
        return

    video_path = sys.argv[1]
    print("Opening:", video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file:", video_path)
        return

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break

        frame = cv2.resize(frame, (640, 480))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        masks = {
            'red': get_color_mask(hsv, 'red'),
            'yellow': get_color_mask(hsv, 'yellow'),
            'green': get_color_mask(hsv, 'green')
        }

        detections = { 'red': [], 'yellow': [], 'green': [] }
        for color_name, mask in masks.items():
            valids = find_valid_contours(mask)
            detections[color_name] = valids
            for d in valids:
                x,y,w,h = d['bbox']
                cv2.rectangle(frame, (x,y), (x+w, y+h), VIS_COLORS[color_name], 2)

        state = classify_state(detections)
        state_color = VIS_COLORS.get(state.lower(), VIS_COLORS['unknown'])
        cv2.putText(frame, f"STATE: {state}", (10,30), font, 0.9, state_color, 2, cv2.LINE_AA)

        cv2.imshow("Traffic Light Detector", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
