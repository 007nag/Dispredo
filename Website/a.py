from PIL import Image
from flask import Flask, render_template, request, redirect
from numpy.lib.utils import source
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import sys
import cv2 
import numpy as np
import os
from keras.models import load_model
models=[]
for i in range(14):
    models.append(load_model("models/"+str(i)+"/"+str(i)+".h5"))
print('Load complete')
app = Flask(__name__)
diseases=["Atelectasis",
"Consolidation",
"Infiltration",
"Pneumothorax",
"Edema",
"Emphysema",
"Fibrosis",
"Effusion",
"Pneumonia",
"Pleural_thickening",
"Cardiomegaly",
"Nodule","Mass",
"Hernia"]
@app.route("/",methods=['POST','GET'])
def zz():
    if request.method=='POST':
        if request.files['image'].filename!='':
            with tf.device("CPU:0"):
                
                #basepath = os.path.dirname(__file__)
                file=request.files['image']
                file_path = os.path.join(
                        'static', 'images', secure_filename(file.filename))
                file.save(file_path)
                try:
                    img = Image.open(file).convert('L')
                    
                    
                    img = np.array(img)
                    img=img.astype(np.float)
                    img = cv2.resize(img,(100,100))
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    img/=255.0
                    
                    img=np.reshape(img,(100,100,1))
                    res=''
                    for i in range(14):
                        z=models[i](np.array([img]),training=False).numpy()[0][0]
                        z=round(z*100,2)
                        if z>=50:
                            res+="<p>There is a high chance for the patient to have <span style='color:green'>"+diseases[i]+"</span></p>"
                    if res=='':
                        res='<p>No diseases have been identified from the X-ray</p>'
                    return render_template("index.html",source=file_path,result=res)
                except:
                    return render_template("index.html",source='static/images/default.png',
                                            result='<p>Please only upload image files</p>')
        return render_template("index.html",source='static/images/default.png',g='')    
    else:
        return render_template("index.html",source='static/images/default.png',g='')
if __name__=="__main__":
    app.run(debug=True)
    