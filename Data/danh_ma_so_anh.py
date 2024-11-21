import os
import glob

def rename_files(duong_dan, duoi_tep, ki_tu_dau, start=1, step=1):
    # Xử lý đường dẫn và đuôi tệp
    pattern = os.path.join(duong_dan, f"*{duoi_tep}")
    
    # Tìm tất cả các tệp trong thư mục có đuôi duoi_tep
    files = glob.glob(pattern)
    
    # Sắp xếp các tệp theo tên (có thể thay đổi nếu muốn sắp xếp theo tiêu chí khác)
    files.sort()

    # Đặt tên lại cho các tệp
    for i, file in enumerate(files):
        # Tính toán chỉ số mới
        new_index = start + i * step
        
        # Lấy tên file và phần mở rộng
        dirname, filename = os.path.split(file)
        name, ext = os.path.splitext(filename)
        
        # Tạo tên mới cho tệp
        new_name = f"{ki_tu_dau}{new_index}{ext}"
        new_path = os.path.join(dirname, new_name)
        
        # Đổi tên tệp
        os.rename(file, new_path)
        print(f"Đổi tên {filename} thành {new_name}")

# Ví dụ sử dụng
duong_dan = r"C:\Users\WIN11\Downloads\Tang_cuong_du_lieu\Car"  # Đường dẫn đến thư mục
duoi_tep = ".png"  # Đuôi tệp cần tìm
ki_tu_dau = "A"  # Ký tự đầu
start = 3001  # Bắt đầu từ chỉ số 1
step = 2   # Bước nhảy là 2

rename_files(duong_dan, duoi_tep, ki_tu_dau, start, step)