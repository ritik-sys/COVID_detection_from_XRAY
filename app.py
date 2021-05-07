from flask import Flask, render_template, redirect, request

import numpy as np
import keras
from keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image

model = keras.models.load_model("model.h5")


# __name__ == __main__
app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def marks():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)  # ./static/images.jpg
        f.save(path)
        img = image.load_img(path, target_size=(256, 256))
    img = np.reshape(img, (1, 256, 256, 3))
    label_map = {'Covid Positive': 0, 'Covid Negative': 1}
    key_list = list(label_map.keys())
    pred = int(model.predict(img)[0][0])
    result_dic = {
        'image': path,
        'caption': key_list[pred]
    }

    return render_template("index.html", your_result=result_dic)


if __name__ == '__main__':
    # app.debug = True
    # due to versions of keras we need to pass another paramter threaded = Flase to this run function
    app.run(debug=False, threaded=False)
