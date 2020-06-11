from flask import Flask, render_template, request, redirect, send_from_directory
from predictor import Predictor
from segmentation import Segmentation
import os
import cv2

app = Flask(__name__,
            static_url_path='', 
            static_folder='./static',
            template_folder='./templates')


@app.route('/', methods=['POST', 'GET'])
def index():
    
    pred = Predictor()
    seg = Segmentation()

    pred.clean()
    seg.clean()

    if request.method == "POST":
        if request.files:
            input_img = request.files["image"].read()
            seg_img = seg.getCropped(input_img)
            pred.getPrediction(input_img)

            cv2.imwrite('./static/proc_imgs/seg_img.jpg', seg_img)

            return render_template('index.html')

    
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)