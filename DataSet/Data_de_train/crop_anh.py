import cv2
import os

def crop_images_in_directory(input_dir, output_dir):
    # Duyệt qua tất cả các tệp trong thư mục đầu vào
    for filename in os.listdir(input_dir):
        # Tạo đường dẫn đầy đủ đến tệp ảnh
        img_path = os.path.join(input_dir, filename)
        
        # Đọc ảnh từ đường dẫn
        image = cv2.imread(img_path)

        height, width, channels = image.shape

        # Tính toán chiều cao mới sau khi cắt 10%
        new_height = int(height * 0.9)

        # Cắt ảnh: lấy phần trên 90% của ảnh
        cropped_image = image[:new_height, :]

        # Lưu ảnh đã cắt vào thư mục đầu ra
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, cropped_image)


# Ví dụ sử dụng
input_directory = 'D:Project_1/New'  # Thay thế bằng thư mục đầu vào của bạn
output_directory = 'D:Project_1/New_1'  # Thay thế bằng thư mục đầu ra của bạn

crop_images_in_directory(input_directory, output_directory)


