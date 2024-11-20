import os
import pandas as pd
import xml.etree.ElementTree as xet
import shutil

from glob import glob
from shutil import copy


#Dẫn đến thư mục chứa thư mục DataSet của chúng ta:
"""Cách 1: Dùng khi tải tập tin về máy, chỉnh sửa lại sao cho phù hợp"""
# folder_Code = os.path.dirname(__file__)
# folder_Project = os.path.dirname(folder_Code)

"""Cách 2: Dùng khi chạy đoạn mã trên GG Colab, chỉnh sửa đường dẫn lại cho phù hợp"""
folder_Project = '/content/drive/MyDrive/OnlineProject'
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

#Sau khi dẫn xong thì bắt đầu chạy.
path = glob(f'{folder_Project}/DataSet/images/*.xml')

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
#####################################
df = pd.DataFrame(labels_dict)
df.to_csv('labels.csv',index=False)
#Các bước này tương tự như khi train model Inception-ResNet-v2


# parsing
def parsing(path):
    parser = xet.parse(path).getroot()
    name = parser.find('filename').text
    filename = f'{folder_Project}/DataSet/images/{name}'

    # width and height
    parser_size = parser.find('size')
    width = int(parser_size.find('width').text)
    height = int(parser_size.find('height').text)

    return filename, width, height
df[['filename','width','height']] = df['filepath'].apply(parsing).apply(pd.Series)

# center_x, center_y, width , height
df['center_x'] = (df['xmax'] + df['xmin'])/(2*df['width'])
df['center_y'] = (df['ymax'] + df['ymin'])/(2*df['height'])

df['bb_width'] = (df['xmax'] - df['xmin'])/df['width']
df['bb_height'] = (df['ymax'] - df['ymin'])/df['height']


### split the data into train and test
df_train = df.iloc[:2000]
df_test = df.iloc[2000:]
#Vì DataSet của chúng ta có 2500 ảnh nên khi chia 80:20 sẽ là 2000 ảnh: 500 ảnh

############

train_folder = 'yolov5/data_images/train'
values = df_train[['filename','center_x','center_y','bb_width','bb_height']].values
for fname, x,y, w, h in values:
    image_name = os.path.split(fname)[-1]
    txt_name = os.path.splitext(image_name)[0]

    dst_image_path = os.path.join(train_folder,image_name)
    dst_label_file = os.path.join(train_folder,txt_name+'.txt')
    print(fname)
    print(dst_image_path)
    # copy each image into the folder
    shutil.copy(fname,dst_image_path)

    # generate .txt which has label info
    label_txt = f'0 {x} {y} {w} {h}'
    with open(dst_label_file,mode='w') as f:
        f.write(label_txt)

        f.close()


test_folder = 'yolov5/data_images/test'
values = df_test[['filename','center_x','center_y','bb_width','bb_height']].values
for fname, x,y, w, h in values:
    image_name = os.path.split(fname)[-1]
    txt_name = os.path.splitext(image_name)[0]

    dst_image_path = os.path.join(test_folder,image_name)
    dst_label_file = os.path.join(test_folder,txt_name+'.txt')

    # copy each image into the folder
    copy(fname,dst_image_path)

    # generate .txt which has label info
    label_txt = f'0 {x} {y} {w} {h}'
    with open(dst_label_file,mode='w') as f:
        f.write(label_txt)

        f.close()