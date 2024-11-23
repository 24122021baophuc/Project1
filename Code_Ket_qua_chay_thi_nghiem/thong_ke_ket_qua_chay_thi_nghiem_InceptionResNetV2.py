import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract as pt
import xml.etree.ElementTree as ET
import os
import re
import Levenshtein as lev  # Thư viện tính toán Edit Distance
import pandas as pd

# Load model
model = tf.keras.models.load_model(r"C:\Users\WIN11\Downloads\my_model (1).keras")  # Thay đổi đường dẫn đến mô hình của bạn

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

# Hàm tính toán IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_pred, y1_pred, x2_pred, y2_pred = box2

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
    return iou

# Hàm tính toán độ chính xác Edit Distance
def calculate_edit_distance(predicted_text, actual_text):
    return lev.distance(predicted_text, actual_text)

# Hàm dự đoán bounding box từ mô hình
def object_detection(path):
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))
    
    image_arr_224 = img_to_array(image1) / 255.0
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    
    coords = model.predict(test_arr)
    
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    return coords[0]

# Hàm nhận diện văn bản biển số xe từ ảnh
def OCR(path, xml_path, txt_path):
    img = np.array(load_img(path))
    
    predicted_coords = object_detection(path)
    
    # Load ground truth bounding box
    ground_truth = load_ground_truth(xml_path)
    if ground_truth is None:
        return None  # Nếu không có bounding box trong XML thì trả về None
    
    with open(txt_path, 'r') as file:
        actual_text = file.read().strip()
    
    # Lấy tọa độ dự đoán và tạo vùng chọn (ROI)
    xmin, xmax, ymin, ymax = predicted_coords
    pre_box = list(map(int, [xmin, ymin, xmax, ymax]))
    roi = img[ymin:ymax, xmin:xmax]
    
    # Chuyển vùng ROI sang ảnh grayscale và cải thiện độ sáng/độ tương phản
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)

    # Sử dụng pytesseract để nhận diện văn bản biển số xe
    text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
    
    # Tối ưu hóa văn bản bằng cách loại bỏ các ký tự không cần thiết
    text = ''.join(re.findall(r'\w+', text))
    
    # Tính toán độ chính xác Edit Distance
    edit_distance = calculate_edit_distance(text, actual_text)
    
    # Tính toán IoU
    iou = calculate_iou(pre_box, ground_truth)
    
    return (pre_box, ground_truth, text, actual_text, edit_distance, iou)

# Hàm điều chỉnh độ sáng và độ tương phản
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

# Hàm xử lý tất cả ảnh trong thư mục và trả về danh sách kết quả
def process_directory(image_dir, xml_dir, txt_dir):
    results = []
    
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg") or image_name.endswith(".png"):
            base_name = os.path.splitext(image_name)[0]  # Lấy tên file mà không có đuôi mở rộng
            
            image_path = os.path.join(image_dir, image_name)
            xml_path = os.path.join(xml_dir, base_name + ".xml")
            txt_path = os.path.join(txt_dir, base_name + ".txt")
            
            if os.path.exists(xml_path) and os.path.exists(txt_path):
                result = OCR(image_path, xml_path, txt_path)
                if result:
                    # Thêm tên file vào kết quả
                    result_with_filename = (base_name,) + result  # Thêm tên file vào đầu tuple
                    results.append(result_with_filename)
    
    return results

# Đường dẫn đến thư mục chứa ảnh, XML và TXT
image_dir = r"C:\Users\WIN11\Downloads\Test"  # Thay đổi đường dẫn đến thư mục ảnh
xml_dir = r"C:\Users\WIN11\Downloads\Test"  # Thay đổi đường dẫn đến thư mục XML
txt_dir = r"C:\Users\WIN11\Downloads\Test"  # Thay đổi đường dẫn đến thư mục TXT

# Tiến hành xử lý và nhận kết quả
results = process_directory(image_dir, xml_dir, txt_dir)

iou_mean = sum([x[-1] for x in results]) / len(results)
edit_distance_mean = sum([x[-2] for x in results]) / len(results)

columns = ['File Name', 'Predicted Box', 'Ground Truth', 'Predicted Text', 'Actual Text', 'Edit Distance', 'IoU']
df = pd.DataFrame(results, columns=columns)

csv_path = r"C:\Users\WIN11\Downloads\result_InceptionResNetV2.csv"  # Đường dẫn lưu tệp CSV
df.to_csv(csv_path, index=False)

print(f'IoU Mean = {iou_mean}')
print(f'Edit Distance Mean = {edit_distance_mean}')



