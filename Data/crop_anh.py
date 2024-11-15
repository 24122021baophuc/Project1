import os
from PIL import Image

def crop_image(input_path, output_path):
    # Mở ảnh
    img = Image.open(input_path)
    
    # Lấy kích thước của ảnh
    width, height = img.size
    
    # Tính toán 10% chiều cao từ dưới
    crop_height = int(height * 0.10)
    
    # Cắt ảnh (loại bỏ 10% chiều cao từ dưới)
    cropped_img = img.crop((0, 0, width, height - crop_height))
    
    # Lưu ảnh đã cắt vào đường dẫn mới
    cropped_img.save(output_path)
    print(f"Đã lưu ảnh đã cắt tại: {output_path}")

def process_images(input_dir, output_dir, file_extension=".jpg"):
    # Kiểm tra nếu thư mục đích không tồn tại, tạo mới
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Duyệt qua tất cả các tệp trong thư mục nguồn
    for filename in os.listdir(input_dir):
        if filename.endswith(file_extension):
            # Xây dựng đường dẫn đầy đủ của ảnh
            input_path = os.path.join(input_dir, filename)
            
            # Tạo tên tệp mới và đường dẫn lưu vào thư mục đích
            output_path = os.path.join(output_dir, filename)
            
            # Gọi hàm để cắt ảnh và lưu vào thư mục đích
            crop_image(input_path, output_path)

# Ví dụ sử dụng
input_dir = "D:/Project1/Data/Anh_Chua_Xu_Ly/Anh_Xe_May"  # Đường dẫn đến thư mục chứa ảnh gốc
output_dir = "D:/Project1/Data/Anh_Chua_Gan_Nhan/Xe_may_s"  # Đường dẫn đến thư mục để lưu ảnh đã xử lý
process_images(input_dir, output_dir, file_extension=".jpg")
