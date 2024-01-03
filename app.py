
from flask import Flask, jsonify, request, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow

import cv2
import base64
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from collections import Counter
import matplotlib.pyplot as plt
from flask_cors import CORS

import io
import os
import datetime

app = Flask(__name__)
CORS(app)
cors = CORS(app, resources={r"/upload_image2": {"origins": "http://localhost:19006"}})
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:''@localhost/flask'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# db = SQLAlchemy(app)
# ma = Marshmallow(app)

# class Articles(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(100))
#     body = db.Column(db.Text())
#     # date = db.Column(db.DateTime, defaut = datetime.datetime.now)
#     def __init__(self, title, body):
#         self.title = title
#         self.body = body

@app.route('/', methods=['GET'])
def get_articles():
    return {"Name": "World"}

@app.route("/api", methods=['GET'])
def parse():
    
    byteImgIO = io.BytesIO()
    byteImg = Image.open("C:/Users/My PC/Downloads/data_trenmang/test/khai-truong-ttkh-cong-hoa-chao-mung-pnj-tron-30-tuoi-4.jpg")
    byteImg.save(byteImgIO, "PNG")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()

    dataBytesIO = io.BytesIO(byteImg)
    Image.open(dataBytesIO)

    return jsonify({"Name": "World"})

@app.route('/upload_image', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data.get('image', None)
    
    if image_data:
        try:
            # Chuyển đổi dữ liệu base64 thành hình ảnh
            img_bytes = base64.b64decode(image_data)
            img = Image.open(BytesIO(img_bytes))

            # Lưu hình ảnh xuống đĩa hoặc thực hiện xử lý trực tiếp
            img.save('received_image.jpg')  # Lưu hình ảnh xuống đĩa
            
            # Hoặc thực hiện xử lý trực tiếp với img
            
            return jsonify({'message': 'Image received and processed successfully!'})
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'})
    else:
        return jsonify({'error': 'No image data received.'})
    
@app.route('/upload_image1', methods=['POST'])
def upload_image1():
    if request.method == 'POST':
        data = request.get_json()
        base64_image = data['base64Image']  # Access the base64 image sent from React Native
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image)

            # Convert image data to PIL image
            img = Image.open(io.BytesIO(image_data))

            # Process the image (optional: save, manipulate, etc.)
            # img.save('saved_image.png')  # Example: Save the image

            return jsonify('Image processed successfully')

        except Exception as e:
            return jsonify(f"Error processing image: {str(e)}")

@app.route('/upload-image2', methods=['POST'])
def upload_image2():
    # try:
    #     image = request.files['image']
    #     pil_image = Image.open(image)
    #     return jsonify('Image processed successfully')
    # except Exception as e:
    #     print('Error:', e)
    #     return 'Error occurred ', 400
    
    f = request.files['upload']
    if f.filename != "":
        # uploaded_file_path = "uploaded/"+f.filename
        # f.save(image_path)
        # age, gender = getAgeGender(f)
        return jsonify('Image processed successfully')




@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        file.save(os.path.join('./uploads', file.filename))  # Save the file to a folder
        return 'File uploaded successfully'







@app.route('/detect', methods=['GET'])
def detect_objects():
    # Load model
    model = YOLO("C:/Users/My PC/Downloads/Train_Data_HH/19_11_23_JoinData_3/train3/weights/best.pt")

    # Load và chuyển đổi hình ảnh
    image = cv2.imread('C:/Users/My PC/Downloads/data_trenmang/test/khai-truong-ttkh-cong-hoa-chao-mung-pnj-tron-30-tuoi-4.jpg')

    # Dự đoán
    results = model.predict(image)

    # Tạo danh sách kết quả để hiển thị
    object_counts = []

    # Lặp qua từng kết quả và đếm số lượng đối tượng được nhận diện cho mỗi lớp
    for result in results:
        names = result.names  # Lấy tên của các lớp
        counts = Counter(result.boxes.cls.tolist())  # Đếm số lượng đối tượng được nhận diện cho mỗi lớp

        # Thêm kết quả vào danh sách để hiển thị
        for class_id, count in counts.items():
            object_counts.append(f"{names[class_id]}: {count} đối tượng")

        # Plot và lưu hình ảnh
        im_array = result.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.show()
        # im.save('media/result3ss.jpg')  # Lưu hình ảnh vào thư mục media của Django

    # Đường dẫn đến hình ảnh đã lưu
    # saved_image_path = 'media/result3ss.jpg'

    # Trả về template với đường dẫn đến hình ảnh
    
    return jsonify({'object_counts': object_counts})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)