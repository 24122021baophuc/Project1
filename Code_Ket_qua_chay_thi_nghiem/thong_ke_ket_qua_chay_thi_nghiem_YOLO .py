import os
import cv2
import numpy as np
import pytesseract as pt
import xml.etree.ElementTree as ET
from skimage import io
import re
import Levenshtein as lev  # Thư viện tính toán Edit Distance
import pandas as pd

# Settings
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Load YOLO Model
net = cv2.dnn.readNetFromONNX(r"D:\Project1\WebbApp_Yolo\Model\weights\best.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Hàm hỗ trợ điều chỉnh độ sáng và độ tương phản
def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

# Hàm trích xuất text từ vùng biển số
def extract_text_from_roi(image, bbox):
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'

    else:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
        text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
        if text != "":
            text = str(text)
            text = text.strip()
            while not text[-1].isalnum(): text = text[:-1]
            while not text[0].isalnum(): text = text[1:]
        return text

# Hàm đọc nhãn từ tệp XML
def load_ground_truth(file_path):
    """
    Load ground truth bounding box from an XML file.
    :param file_path: Path to the XML file containing the bounding box information.
    :return: Ground truth bounding box as a tuple (xmin, xmax, ymin, ymax)
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Hoán đổi ymin và ymax
        return (xmin, ymin, xmax, ymax) 
    
    return None
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_pred, y1_pred, w, h = box2
    x2_pred = x1_pred + w
    y2_pred = y1_pred + h

    # Tính diện tích của từng bounding box
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_pred - x1_pred) * (y2_pred - y1_pred)

    # Tính giao giữa hai bounding box
    inter_x1 = max(x1, x1_pred)
    inter_y1 = max(y1, y1_pred)
    inter_x2 = min(x2, x2_pred)
    inter_y2 = min(y2, y2_pred)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    # Tính toán IoU
    union_area = area1 + area2 - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou, (x1_pred, y1_pred, x2_pred, y2_pred)

# Hàm dự đoán bounding box và text từ ảnh
def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

# Hàm lọc các kết quả dự đoán
def non_maximum_supression(input_image, detections):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        
        if confidence > 0.4:
            class_score = row[5]
            if class_score > 0.25:  # Ban đầu 0.25
                cx, cy, w, h = row[0:4]

                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                confidences.append(confidence)
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, 0.25, 0.45)
    return boxes_np, confidences_np, index

# Hàm đọc nội dung từ tệp văn bản thực tế
def read_text_from_file(text_file_path):
    if os.path.exists(text_file_path):
        with open(text_file_path, 'r') as file:
            return file.read().strip()
    else:
        return 'No text file found'

def calculate_edit_distance(predicted_text, actual_text):
    return lev.distance(predicted_text, actual_text)

# Hàm xử lý tệp ảnh và XML trong thư mục
def process_images_and_xml(folder_path):
    # Lấy tất cả các tệp ảnh và XML trong thư mục
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

    result = []

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        xml_path = os.path.join(folder_path, image_file.replace('.jpg', '.xml').replace('.png', '.xml'))
        text_file_path = os.path.join(folder_path, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))


        # Dự đoán và trích xuất biển số
        img = io.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        input_image, detections = get_detections(img, net)
        boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)

        # Đọc chuỗi văn bản thực tế từ tệp .txt
        real_text = read_text_from_file(text_file_path)
        # Lưu các kết quả
        if len(boxes_np) > 0:
            # Lấy bounding box thực tế từ XML (trong trường hợp có nhiều box, ta so sánh với các box thực tế)           
            ground_truth = load_ground_truth(xml_path)
            if ground_truth is None:
                return None  # Nếu không có bounding box trong XML thì trả về None
            
            predicted_box = boxes_np[0]
            iou, predicted_box = calculate_iou(ground_truth, predicted_box)
            predicted_text = extract_text_from_roi(img, predicted_box)
            predicted_text = ''.join(re.findall(r'[A-Za-z0-9]+', predicted_text))
            edit_distance = calculate_edit_distance(real_text, predicted_text)

            # Lưu kết quả vào danh sách kết quả
            result.append((os.path.splitext(image_file)[0], predicted_box, ground_truth, predicted_text, real_text, edit_distance, iou ))

    return result

# Đường dẫn đến thư mục chứa ảnh, XML và văn bản
folder_path = r'C:\Users\WIN11\Downloads\Test'

# Gọi hàm và in kết quả
results = process_images_and_xml(folder_path)

iou_mean = sum([x[-1] for x in results]) / len(results)
edit_distance_mean = sum([x[-2] for x in results]) / len(results)

columns = ['File Name', 'Predicted Box', 'Ground Truth', 'Predicted Text', 'Actual Text', 'Edit Distance', 'IoU']
df = pd.DataFrame(results, columns=columns)

csv_path = r"C:\Users\WIN11\Downloads\result_YOLO.csv"  # Đường dẫn lưu tệp CSV
df.to_csv(csv_path, index=False)

print(f'IoU Mean = {iou_mean}')
print(f'Edit Distance Mean = {edit_distance_mean}')
