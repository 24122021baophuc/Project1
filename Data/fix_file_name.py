import os
import xml.etree.ElementTree as ET

def update_xml_filenames(directory):
    # Lấy danh sách tất cả các file trong thư mục
    files = os.listdir(directory)

    # Lọc ra file XML và các file ảnh (đuôi jpg, png)
    xml_files = [f for f in files if f.endswith('.xml')]
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    # Xử lý từng file XML
    for xml_file in xml_files:
        # Lấy tên gốc (không bao gồm phần đuôi)
        base_name = os.path.splitext(xml_file)[0]

        # Tìm file ảnh khớp với phần tên gốc
        matching_image = next((img for img in image_files if os.path.splitext(img)[0] == base_name), None)

        if matching_image:
            # Đường dẫn tới file XML
            xml_path = os.path.join(directory, xml_file)

            # Đọc nội dung file XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Tìm và cập nhật thẻ <filename>
            filename_element = root.find('filename')
            if filename_element is not None:
                filename_element.text = matching_image

            # Ghi lại file XML, không có khai báo
            with open(xml_path, 'wb') as xml_out:
                tree.write(xml_out, encoding='utf-8', xml_declaration=False)
            print(f"Updated '{xml_file}' with filename '{matching_image}'")
        else:
            print(f"No matching image found for '{xml_file}'")

# Chạy chương trình
directory_path = "D:\Project1\Data\Train\Train_3"  # Thay bằng đường dẫn tới thư mục của bạn
update_xml_filenames(directory_path)
