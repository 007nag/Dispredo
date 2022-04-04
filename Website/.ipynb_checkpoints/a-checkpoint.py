from PIL import Image
from flask import Flask, render_template, request, redirect
from tensorflow import keras
import cv2 
import numpy as np
import os
app = Flask(__name__)

@app.route("/")
def zz():
    return render_template("index.html")
@app.route("/upload",methods=['POST','GET'])
def upload():
    if request.method=='POST':
        if request.files:
            print('ok')
            img = Image.open(request.files['image']).convert('L')
            img = np.array(img)
            img=img.astype(np.float)
            img = cv2.resize(img,(100,100))
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img/=255.0
            
            img=np.reshape(img,(100,100,1))
            
            """filestr = request.files['image'].read()
            npimg = np.fromstring(filestr, np.uint8)
            npimg=npimg.astype(np.float64)
            img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)"""
            model= keras.models.load_model("models/pneu1.h5")
            z=model.predict(np.array([img]))
            print(z)
            return str(z[0][0])
            #return "!!!"
    else:
        return render_template("upload.html")
if __name__=="__main__":
    app.run(debug=True)
    