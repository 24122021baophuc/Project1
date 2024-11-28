import cv2
import numpy as np
import pytesseract as pt
from skimage import io

"""Sau khi chạy chương trình thì bấm ESC để tắt.
Trước khi chạy hãy thay đổi đường dẫn đến mô hình YOLO"""

# url = 'http://192.168.1.19:8080/video'
# video_cap = cv2.VideoCapture(url)
# Ở trên là dùng ứng dụng điện thoại "IP Webcam" để biến camera điện thoại thành camera của máy tính.

video_cap = cv2.VideoCapture(0)
# Ở đây, camera là camera của máy tính, tùy vào máy mà chỉ số khác nhau(0, 1, ...), với máy chạy thí nghiệm thì camera có chỉ số 0.

# settings
INPUT_WIDTH =  640
INPUT_HEIGHT = 640


# LOAD YOLO MODEL - Load mô hình:
net = cv2.dnn.readNetFromONNX('C:/Users/Admin/Documents/Project1/WebbApp_Yolo/Model/weights/best.onnx') # Đường dẫn đến mô hình.
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf

def extract_text(image,bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'

    else:
        roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        magic_color = apply_brightness_contrast(gray, brightness=40, contrast=70)
        # text = pt.image_to_string(magic_color) 
        text = pt.image_to_string(magic_color, lang='eng', config='--psm 6')
        print(text)
        if text != "":
            text = str(text)
            text = text.strip()
            try:
                while not text[-1].isalnum(): text = text[:-1]
                while not text[0].isalnum(): text = text[1:]
            except:
                pass
        return text

def get_detections(img,net):
    # 1.CONVERT IMAGE TO YOLO FORMAT - Định dạng mô hình:
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row,col)
    input_image = np.zeros((max_rc,max_rc,3),dtype=np.uint8)
    input_image[0:row,0:col] = image

    # 2. GET PREDICTION FROM YOLO MODEL - Dự đoán kết quả:
    blob = cv2.dnn.blobFromImage(input_image,1/255,(INPUT_WIDTH,INPUT_HEIGHT),swapRB=True,crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]
    return input_image, detections

def non_maximum_supression(input_image,detections):
    # 3. FILTER DETECTIONS BASED ON CONFIDENCE AND PROBABILIY SCORE - Lọc kết quả:

    # center x, center y, w , h, conf, proba
    boxes = []
    confidences = []
    
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w/INPUT_WIDTH
    y_factor = image_h/INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4] # confidence of detecting license plate
        
        if confidence > 0.4:
            class_score = row[5] # probability score of license plate
            if class_score > 0.25: ###Ban đầu 0.25
                cx, cy , w, h = row[0:4]

                left = int((cx - 0.5*w)*x_factor)
                top = int((cy-0.5*h)*y_factor)
                width = int(w*x_factor)
                height = int(h*y_factor)
                box = np.array([left,top,width,height])
                confidences.append(confidence)
                boxes.append(box)

    # 4.1 CLEAN
    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()

    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45) #0.5,0.45
    return boxes_np, confidences_np, index

# predictions flow with return result
def yolo_predictions(img,net):
    # step-1: detections
    input_image, detections = get_detections(img,net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    return boxes_np, confidences_np, index

while True:
    _, frame = video_cap.read()
    boxes_np, confidences_np, index = yolo_predictions(frame, net)
    for ind in index:
        x,y,w,h =  boxes_np[ind]
        
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        license_text = extract_text(frame,boxes_np[ind])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0, 255, 0),2)
        cv2.putText(frame,conf_text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        cv2.putText(frame,license_text,(x,y+h+27),cv2.FONT_HERSHEY_SIMPLEX,0.6,(225,0,0),2)
    cv2.imshow("Camera", frame)
#------------------------------

    if cv2.waitKey(1) & 0xff == 27: # Bấm ESC để tắt.
        break

video_cap.release()
cv2.destroyAllWindows()