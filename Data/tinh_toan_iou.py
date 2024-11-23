import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import xml.etree.ElementTree as ET
import pytesseract as pt

# Load mô hình học sâu đã huấn luyện
model = tf.keras.models.load_model(r"C:\Users\WIN11\Downloads\my_model (1).keras")

# Đọc nhãn từ file XML (PASCAL VOC)
def read_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes

# Tính toán IoU giữa hai hộp (bounding boxes)
def calculate_iou(pred_box, true_box):
    x1_pred, y1_pred, x2_pred, y2_pred = pred_box
    x1_true, y1_true, x2_true, y2_true = true_box
    
    # Tính diện tích giao nhau
    x1_inter = max(x1_pred, x1_true)
    y1_inter = max(y1_pred, y1_true)
    x2_inter = min(x2_pred, x2_true)
    y2_inter = min(y2_pred, y2_true)
    
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    
    # Tính diện tích hợp nhất
    pred_area = (x2_pred - x1_pred) * (y2_pred - y1_pred)
    true_area = (x2_true - x1_true) * (y2_true - y1_true)
    
    union_area = pred_area + true_area - inter_area
    
    iou = inter_area / union_area
    return iou

# Dự đoán và tính IoU cho một ảnh
def process_image(image_path, xml_path, filename):
    image = load_img(image_path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8 bit array (0,255)
    image_resized = load_img(image_path, target_size=(224, 224))  # Resize image to match model input
    image_arr_224 = img_to_array(image_resized) / 255.0  # Normalize
    h, w, d = image.shape  # Image dimensions
    test_arr = image_arr_224.reshape(1, 224, 224, 3)  # Reshape for model input

    # Dự đoán tọa độ của bounding box
    coords = model.predict(test_arr)
    
    # Denormalize coordinates to match the original image dimensions
    denorm = np.array([w, w, h, h])  # Denormalization factors for image width and height
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    # Bounding box predicted
    xmin_pred, xmax_pred, ymin_pred, ymax_pred = coords[0]
    predicted_box = (xmin_pred, ymin_pred, xmax_pred, ymax_pred)
    
    # Đọc nhãn thực tế từ XML
    true_boxes = read_xml(xml_path)
    
    # Tính toán IoU cho mỗi bounding box thực tế
    iou_scores = []
    for true_box in true_boxes:
        iou = calculate_iou(predicted_box, true_box)
        iou_scores.append(iou)
    
    # Lọc những IoU lớn hơn ngưỡng (ví dụ: 0.5)
    iou_threshold = 0.5
    valid_iou = [iou for iou in iou_scores if iou >= iou_threshold]
    
    print(f"IOU Scores for {filename}: {iou_scores}")
    print(f"Valid IOU scores (threshold > {iou_threshold}): {valid_iou}")

    return iou_scores, valid_iou

# Xử lý tất cả ảnh trong thư mục và tính toán IoU
def process_directory(image_dir, xml_dir):
    li = []
    image_files = os.listdir(image_dir)
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            xml_path = os.path.join(xml_dir, image_file.replace('.jpg', '.xml').replace('.png', '.xml').replace('.jpeg', '.xml'))
            
            if os.path.exists(xml_path):
                print(f"Processing image: {image_file}")
                iou_scores, valid_iou = process_image(image_path, xml_path, image_file)
                li.append(iou_scores[0])
                
                # Có thể lưu kết quả vào file hoặc cơ sở dữ liệu tùy theo yêu cầu
            else:
                print(f"XML file not found for {image_file}")
    return sum(li) / len(li)

# Ví dụ sử dụng
image_dir = r'C:\Users\WIN11\Downloads\images-20241123T025304Z-001\Data_new'  # Đường dẫn thư mục chứa ảnh
xml_dir = r'C:\Users\WIN11\Downloads\images-20241123T025304Z-001\Data_new'  # Đường dẫn thư mục chứa các file XML

print(process_directory(image_dir, xml_dir))

