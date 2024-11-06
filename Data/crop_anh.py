import os
from PIL import Image

# Đường dẫn đến thư mục chứa ảnh gốc và thư mục lưu ảnh sau khi cắt
input_folder = 'D:/Project1/Data/Anh_Chua_Xu_Ly/Anh_Xe_May'
output_folder = 'D:/Project1/Data/Anh_Chua_Gan_Nhan/Anh_Xe_May_1'

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Tỉ lệ cắt bỏ phần dưới (phần chứa thời gian)
remove_ratio = 0.1  # Ví dụ: bỏ đi 10% chiều cao ở dưới

# Duyệt qua từng file trong thư mục
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Mở ảnh
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        width, height = img.size

        # Bỏ phần thời gian ở bên dưới
        crop_height = int(height * (1 - remove_ratio))
        img_cropped = img.crop((0, 0, width, crop_height))

        # Tính toán chiều rộng và chiều cao mới để đạt tỉ lệ 4:3
        new_width = width
        new_height = int(width * 3 / 4)
        
        # Nếu chiều cao của ảnh sau khi bỏ thời gian không đủ 4:3, lấy theo chiều cao
        if new_height > crop_height:
            new_height = crop_height
            new_width = int(crop_height * 4 / 3)

        # Cắt ảnh theo tỉ lệ 4:3
        left = (width - new_width) / 2
        top = (crop_height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (crop_height + new_height) / 2
        img_final = img_cropped.crop((left, top, right, bottom))

        # Lưu ảnh đã cắt vào thư mục đích
        output_path = os.path.join(output_folder, filename)
        img_final.save(output_path)

print("Hoàn thành cắt ảnh hàng loạt.")
