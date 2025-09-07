import cv2
import numpy as np
import sys
import math

# Improved color ranges with better HSV values
COLOR_RANGES = {
    'red1':   ((0, 50, 50),   (10, 255, 255)),    # Lower red range
    'red2':   ((170, 50, 50), (180, 255, 255)),   # Upper red range
    'yellow': ((20, 100, 100), (30, 255, 255)),   # More specific yellow
    'green':  ((40, 50, 50),   (80, 255, 255)),   # Better green range
}

VIS_COLORS = {
 'red': (0, 0, 255),
 'yellow': (0, 255, 255), 
 'green': (0, 255, 0),
 'unknown': (128, 128, 128)
}

def preprocess_frame(frame):
 blurred = cv2.GaussianBlur(frame, (5, 5), 0)
 lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
 l, a, b = cv2.split(lab)
 clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
 l = clahe.apply(l)
 enhanced = cv2.merge([l, a, b])
 enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
 return enhanced

def get_color_mask(hsv, color_name):
 if color_name == 'red':
  lower1, upper1 = COLOR_RANGES['red1']
  lower2, upper2 = COLOR_RANGES['red2']
  mask1 = cv2.inRange(hsv, np.array(lower1), np.array(upper1))
  mask2 = cv2.inRange(hsv, np.array(lower2), np.array(upper2))
  mask = cv2.bitwise_or(mask1, mask2)
 else:
  lower, upper = COLOR_RANGES[color_name]
  mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
 kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
 mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
 mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
 return mask

def is_circular(contour, min_circularity=0.3):
 area = cv2.contourArea(contour)
 if area < 100:
  return False
 perimeter = cv2.arcLength(contour, True)
 if perimeter == 0:
  return False
 circularity = 4 * math.pi * area / (perimeter * perimeter)
 return circularity >= min_circularity

def find_valid_contours(mask, min_area=150, max_area=5000):
 contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 valid = []
 for contour in contours:
  area = cv2.contourArea(contour)
  if area < min_area or area > max_area:
   continue
  if not is_circular(contour):
   continue
  x, y, w, h = cv2.boundingRect(contour)
  aspect_ratio = w / h if h > 0 else 0
  if not (0.5 <= aspect_ratio <= 2.0):
   continue
  center_x = x + w // 2
  center_y = y + h // 2
  valid.append({
   'area': area,
   'bbox': (x, y, w, h),
   'center': (center_x, center_y),
   'contour': contour
  })
 return valid

def group_traffic_lights(all_detections):
 all_lights = []
 for color, detections in all_detections.items():
  for detection in detections:
   detection['color'] = color
   all_lights.append(detection)
 if not all_lights:
  return []
 all_lights.sort(key=lambda x: x['center'][1])
 groups = []
 used = set()
 for i, light in enumerate(all_lights):
  if i in used:
   continue
  group = [light]
  used.add(i)
  for j, other_light in enumerate(all_lights):
   if j in used or i == j:
    continue
   x_diff = abs(light['center'][0] - other_light['center'][0])
   y_diff = abs(light['center'][1] - other_light['center'][1])
   if x_diff < 50 and 30 < y_diff < 200:
    group.append(other_light)
    used.add(j)
  if group:
   groups.append(group)
 return groups

def classify_traffic_light_state(groups):
 if not groups:
  return 'UNKNOWN'
 best_group = max(groups, key=lambda g: sum(d['area'] for d in g))
 color_counts = {'red': 0, 'yellow': 0, 'green': 0}
 color_areas = {'red': 0, 'yellow': 0, 'green': 0}
 for detection in best_group:
  color = detection['color']
  color_counts[color] += 1
  color_areas[color] += detection['area']
 if color_areas['red'] > 0 and color_areas['red'] >= max(color_areas['yellow'], color_areas['green']):
  return 'RED'
 elif color_areas['green'] > 0 and color_areas['green'] >= max(color_areas['red'], color_areas['yellow']):
  return 'GREEN'  
 elif color_areas['yellow'] > 0:
  return 'YELLOW'
 elif color_areas['red'] > 0:
  return 'RED'
 else:
  return 'UNKNOWN'

def draw_detections(frame, groups, state):
 font = cv2.FONT_HERSHEY_SIMPLEX
 for group in groups:
  for detection in group:
   x, y, w, h = detection['bbox']
   color = detection['color']
   vis_color = VIS_COLORS[color]
   cv2.rectangle(frame, (x, y), (x + w, y + h), vis_color, 2)
   center = detection['center']
   cv2.circle(frame, center, 3, vis_color, -1)
   cv2.putText(frame, color.upper(), (x, y - 5), font, 0.4, vis_color, 1)
 state_color = VIS_COLORS.get(state.lower(), VIS_COLORS['unknown'])
 cv2.putText(frame, f"TRAFFIC LIGHT: {state}", (10, 30), font, 0.8, state_color, 2)
 total_detections = sum(len(group) for group in groups)
 cv2.putText(frame, f"Detections: {total_detections}", (10, 60), font, 0.6, (255, 255, 255), 1)

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
 state_history = []
 history_length = 5
 print("Press 'q' to quit, 's' to save current frame")
 while True:
  ret, frame = cap.read()
  if not ret:
   print("End of video or cannot read frame")
   break
  frame = cv2.resize(frame, (640, 480))
  processed = preprocess_frame(frame)
  hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
  masks = {
   'red': get_color_mask(hsv, 'red'),
   'yellow': get_color_mask(hsv, 'yellow'),
   'green': get_color_mask(hsv, 'green')
  }
  detections = {}
  for color_name, mask in masks.items():
   detections[color_name] = find_valid_contours(mask)
  groups = group_traffic_lights(detections)
  current_state = classify_traffic_light_state(groups)
  state_history.append(current_state)
  if len(state_history) > history_length:
   state_history.pop(0)
  if state_history:
   state_counts = {state: state_history.count(state) for state in set(state_history)}
   final_state = max(state_counts.items(), key=lambda x: x[1])[0]
  else:
   final_state = current_state
  draw_detections(frame, groups, final_state)
  cv2.imshow("Traffic Light Detector", frame)
  key = cv2.waitKey(30) & 0xFF
  if key == ord('q'):
   break
  elif key == ord('s'):
   cv2.imwrite(f"traffic_light_detection.jpg", frame)
   print("Frame saved!")
 cap.release()
 cv2.destroyAllWindows()

if __name__ == "__main__":
 main()
