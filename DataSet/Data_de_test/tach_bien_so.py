import os
import glob

def rename_files(duong_dan, ten_text, thu_muc_con, ki_tu_dau, start=1, step=1):
    #Đọc file:
    input_path = os.path.join(duong_dan, ten_text)
    with open(input_path, 'r', encoding='utf-8') as file:
        bien_so = [line.strip() for line in file.readlines() if line != '\n']
    # Tạo các tệp:
    for i in range(len(bien_so)):
        # Tính toán chỉ số mới
        new_index = start + i * step
        new_index = str(new_index).zfill(3)
        
        #Tạo tên file và tạo thư mục con
        file_name = ki_tu_dau + new_index + '.txt'
        file_path = os.path.join(duong_dan, thu_muc_con)
        try:
            os.makedirs(file_path,exist_ok=0)
        except:
            pass
        file_path = os.path.join(file_path, file_name)
        # Mở file ở chế độ ghi ('w') và tạo file:
        with open(file_path, 'w', encoding='utf-8') as file_ghi:
            file_ghi.write(bien_so[i])

# Ví dụ sử dụng
duong_dan = r"C:\Users\Admin\Documents\Project1\Data_de_test"  # Đường dẫn đến thư mục chưa file text lưu biển số
thu_muc_con = r"Anh_va_file" #Tên thư mục con muốn tạo
ten_text = r"Bien_so_test.txt" #Tên file text lưu biển số
ki_tu_dau = "Test_"  # Ký tự đầu
start = 1  # Bắt đầu từ chỉ số 1
step = 1   # Bước nhảy là 2

rename_files(duong_dan, ten_text, thu_muc_con, ki_tu_dau, start, step)