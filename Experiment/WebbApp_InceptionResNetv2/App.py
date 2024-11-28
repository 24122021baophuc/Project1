from flask import Flask, render_template, request
import os
from deeplearning import OCR
import glob
# webserver gateway interface
app = Flask(__name__)

BASE_PATH = os.path.dirname(__file__)
UPLOAD_PATH = BASE_PATH + '/static/upload'

def clear_folders():
    # Danh sách các thư mục cần xóa ảnh
    folders = ["static/upload", "static/predict", "static/roi"]
    for folder in folders:
        os.makedirs(f"{BASE_PATH}/{folder}", exist_ok=1)
        files = glob.glob(f"{BASE_PATH}/{folder}/*")  # Lấy tất cả file trong thư mục
        for file in files:
            try:
                os.remove(file)  # Xóa từng file
            except Exception as e:
                print(f"Không thể xóa file {file}: {e}")

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        clear_folders()
        upload_file = request.files['image_name']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text = OCR(path_save, filename)

        return render_template('index.html', upload=True, upload_image=filename, text=text)

    return render_template('index.html', upload=False)


if __name__ == "__main__":
    app.run(debug=True)