##############################Phần 2.4:################################
# a) Cài đặt các thư viện cần thiết

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import pytesseract as pt
import plotly.express as px
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xet

from glob import glob
from skimage import io
from shutil import copy
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# b) Dẫn đến thư mục chứa DataSet:
"""Cách 1: Dùng đường dẫn tương đối để làm việc trên máy tính cá nhân
# GGColab dùng GG Drive, chúng ta sẽ dùng trực tiếp từ thư mục thông qua đường dẫn tương đối:
# Với vị trí các thư mục như hiện tại, cách làm sẽ như sau:
Folder_Code = os.path.dirname(__file__)
# Folder_code sẽ lưu đường dẫn đến folder "Code".

Folder_Project = os.path.dirname(Folder_Code)
# Folder_Project sẽ lưu đường dẫn đến folder "Project"
path = glob(f'{Folder_Project}/DataSet/images/*.xml')
#Dùng hàm glob() để tạo ra 1 list "path" lưu địa chỉ dẫn đến các file '.xml' lưu trong dataset."""

"""Cách 2: Dùng GG Colbab để chạy file trên GG Colab"""
Folder_Project = '/content/drive/MyDrive/Project'
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
path = glob(f'{Folder_Project}/DataSet/images/*.xml')
#Dùng hàm glob() để tạo ra 1 list "path" lưu địa chỉ dẫn đến các file '.xml' lưu trong dataset.

labels_dict = dict(filepath=[],xmin=[],xmax=[],ymin=[],ymax=[])
for i in path:
    info = xet.parse(i)
    root = info.getroot()
    member_object = root.find('object')
    labels_info = member_object.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(i)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)
df = pd.DataFrame(labels_dict)
df.to_csv(f'{Folder_Project}/labels.csv',index=False)
#Các câu lệnh trên có tác dụng xử lý các file '.xml' của chúng ta, lấy ra các giá trị xmin max, ymin max và lưu trúng vào file
#'.csv' (Một dạng file tương đồng với Excel)

"""

# file_path = image_path[38] #path of our image N137.jpeg
# img = cv2.imread(file_path) #read the image
# # xmin-1804/ymin-1734/xmax-2493/ymax-1882
# img = io.imread(file_path) #Read the image
# fig = px.imshow(img)
# fig.update_layout(width=600, height=500, margin=dict(l=10, r=10, b=10, t=10),xaxis_title='Figure 8 - N137.jpeg with bounding box')
# fig.add_shape(type='rect',x0=401, x1=593, y0=456, y1=493, xref='x', yref='y',line_color='cyan')
# fig.show()

"""
#Đoạn code được ẩn phía trên là nháp, có tác dụng hiện ảnh ra sử dụng các thư viện như OpenCV, và vẽ 1 hình chữ nhật đánh dấu
#lên ảnh xem thử file '.xml' có được đọc đúng hay không. Nói chung là không cần thiết


#c) Xử lý Dataset
def getFilename(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join(f'{Folder_Project}/DataSet/images',filename_image)
    return filepath_image
image_path = list(df['filepath'].apply(getFilename))
#Tạo ra list "image_path" lưu trữ đường dẫn đến các file ảnh trong Dataset




###################Phần 3:######################
labels = df.iloc[ : , 1 : ].values
#Tạo ra "labels" lưu các giá trị xmax min, ymax min tương ứng của từng ảnh
data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind]
    img_arr = cv2.imread(image)
    h,w,d = img_arr.shape
    # Prepprocesing
    load_image = load_img(image,target_size=(224,224))
    load_image_arr = img_to_array(load_image)
    norm_load_image_arr = load_image_arr/255.0 # Normalization
    # Normalization to labels
    xmin,xmax,ymin,ymax = labels[ind]
    nxmin,nxmax = xmin/w,xmax/w
    nymin,nymax = ymin/h,ymax/h
    label_norm = (nxmin,nxmax,nymin,nymax) # Normalized output
    # Append
    data.append(norm_load_image_arr)
    output.append(label_norm)
#Đoạn code trên có tác dụng lấy ra từng ảnh và chuyển chúng sang dạng array bằng cách sử dụng OpenCV, ngoài ra còn resize
#lại ảnh sang 224x224 để phù hợp cho quá trình tạo model. Sau khi resize, chúng ta thực hiện bước Normalization bằng cách
#chia ảnh cho 225 = 2 mũ 8 - 1. Chúng ta Normaliize cả ảnh gốc và phần label. Sau cùng ta được kết quả là kích thước ảnh ở
#khoảng từ 0 đến 1, phù hợp cho model học sâu.


X = np.array(data,dtype=np.float32)
y = np.array(output,dtype=np.float32)
print(X.shape, y.shape)
#2 câu lệnh trên gọi là quá trình tiền xử lý chuẩn bị:
#Chuyển đổi dữ liệu dạng list ở trên sang Numpy. Việc chuyển dữ liệu sang dạng mảng Numpy đảm bảo rằng dữ liệu được định dạng
#chính xác và tối ưu hóa cho quá trình tính toán trong các framework học sâu như TensorFlow hoặc PyTorch.

x_train,x_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=0)
#Câu lệnh trên có tác dụng chia tập dữ liệu thành 2 tập: TẬP HUẤN LUYỆN và TẬP KIỂM TRA:
# + Tập huấn luyện: Mô hình được huấn luyện trên tập huấn luyện
# + Tập kiểm tra: Mô hình sau khi được huấn luyện sẽ được đánh giá trên tập kiểm tra
# "train_size=0.8" có nghĩa là 80% dữ liệu từ Dataset sẽ được dùng để huấn luyện, còn lại là để kiểm tra.



##################################Phần 4:##################################
##4.1) Xây dựng model:
#Giới thiệu sơ bộ về mô hình InceptionResNetV2:
#InceptionResNetV2 là mô hình đã được huấn luyện trước trên tập dữ liệu ImageNet, chúng ta sẽ sử dụng nó để xây dựng
#mô hình học sâu cần thiết.
inception_resnet = InceptionResNetV2(weights="imagenet",include_top=False, input_tensor=Input(shape=(224,224,3)))
# ---------------------
headmodel = inception_resnet.output
headmodel = Flatten()(headmodel)
headmodel = Dense(500,activation="relu")(headmodel)
headmodel = Dense(250,activation="relu")(headmodel)
headmodel = Dense(4,activation='sigmoid')(headmodel)

# ---------- Tạo model
model = Model(inputs=inception_resnet.input,outputs=headmodel)

# Complie model - Biên dịch Model
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))

"""model.summary()"""
#Câu lệnh giúp hiển thị kiến trúc của mô hình, có thể bật lên để xem thử, không cần thiết.



##4.2)
#Trong thư viện TensorFlow, sử dụng TensorBoard, cụ thể ra sao thì chịu, đọc không hiểu =))
log_dir = os.path.abspath(f"{Folder_Project}/object_detection/train")
os.makedirs(log_dir, exist_ok=True)

tfb = TensorBoard(log_dir)
history = model.fit(x=x_train,y=y_train,batch_size=10,epochs=100,validation_data=(x_test,y_test),callbacks=[tfb])
model.save(f'{Folder_Project}/object_detection/train/my_model.keras')
#Huấn luyện và lưu lại mô hình