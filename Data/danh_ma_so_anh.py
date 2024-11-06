import os

def rename_files_odd_num(directory, prefix):
    # Lấy danh sách các file trong thư mục
    files = os.listdir(directory)
    
    odd_number = 1  # Bắt đầu với số lẻ 1
    
    for file in files:
        # Tạo đường dẫn đầy đủ cho file
        old_file_path = os.path.join(directory, file)
        
        # Kiểm tra xem có phải là file không
        if os.path.isfile(old_file_path):
            # Tạo tên mới với số lẻ
            new_file_name = f"{prefix}{odd_number}{os.path.splitext(file)[1]}"
            new_file_path = os.path.join(directory, new_file_name)
            
            # Đổi tên file
            os.rename(old_file_path, new_file_path)
            print(f"Đã đổi tên: {old_file_path} -> {new_file_path}")
            
            # Cập nhật số lẻ cho file tiếp theo
            odd_number += 2
            
def rename_files_even_num(directory, prefix):
    # Lấy danh sách các file trong thư mục
    files = os.listdir(directory)
    
    even_number = 2  # Bắt đầu với số chẵn 2
    
    for file in files:
        # Tạo đường dẫn đầy đủ cho file
        old_file_path = os.path.join(directory, file)
        
        # Kiểm tra xem có phải là file không
        if os.path.isfile(old_file_path):
            # Tạo tên mới với số chẵn
            new_file_name = f"{prefix}{even_number}{os.path.splitext(file)[1]}"
            new_file_path = os.path.join(directory, new_file_name)
            
            # Đổi tên file
            os.rename(old_file_path, new_file_path)
            print(f"Đã đổi tên: {old_file_path} -> {new_file_path}")
            
            # Cập nhật số chẵn cho file tiếp theo
            even_number += 2

# Sử dụng hàm đánh mã số lẻ cho ảnh xe ô tô
directory = "D:/Project1/Data/Anh_Chua_Gan_Nhan/Anh_Car_1"
prefix = "A"
rename_files_odd_num(directory, prefix)

# Sử dụng hàm đánh mã số chẵn cho ảnh xe máy
directory = "D:/Project1/Data/Anh_Chua_Gan_Nhan/Anh_Xe_May_1"
prefix = "A"
rename_files_even_num(directory, prefix)