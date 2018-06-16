
# coding: utf-8

# In[1]:


import os
import subprocess
import numpy as np
import json
import model as EnsembleModel
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from feature_extractor import feature_extractor as fe
from graphviz import Digraph
from sklearn import preprocessing
import pandas as pd

feat = pd.read_csv("FMA_8000_with_track_and_genres.csv")
feat = feat.drop("Unnamed: 0",axis=1)
cols = feat.iloc[:,160:].columns
feat = feat.drop(cols,axis=1).as_matrix()

# In[2]:
global CONSIDER_DATASET
global classification_done
global result
global files
CONSIDER_DATASET = True
classification_done = False
result = []
files = []
path_dictionary = {
    "Pop" : ["Folk_International_Pop_Rock","Pop_Rock" ],
    "Rock": ["Folk_International_Pop_Rock","Pop_Rock" ],
    "Folk": ["Folk_International_Pop_Rock", "Folk_International"],
    "International" : ["Folk_International_Pop_Rock", "Folk_International"],
    "Electronic" : ["Electronic_Experimental_Hip-Hop_Instrumental", "Electronic_Experimental"],
    "Experimental" : ["Electronic_Experimental_Hip-Hop_Instrumental", "Electronic_Experimental"],
    "Hip-Hop" : ["Electronic_Experimental_Hip-Hop_Instrumental", "Hip-Hop_Instrumental"],
    "Instrumental" : ["Electronic_Experimental_Hip-Hop_Instrumental", "Hip-Hop_Instrumental"]
}
UPLOAD_FOLDER = './static/uploads'
#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif',"mp3"])
ALLOWED_EXTENSIONS = set(["mp3","wav"])
TMP_FOLDER = './static/tmp'

app = Flask(__name__)#,template_folder=".",static_url_path='')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload.html', methods=['GET','POST'])
def upload_file():
    global classification_done
    print(classification_done)
    if request.method == 'POST':

        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("upload.html",obj={"err":"Please choose a file"})
        file = request.files['file']

        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return render_template("upload.html",obj={"err":"Please choose a file"})

        if not allowed_file(file.filename):
            return render_template("upload.html",obj={"err":"Error : Unrecognised music file"})
        if file and allowed_file(file.filename):

            filename = secure_filename("_".join(file.filename.split()))
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #return redirect(url_for('uploaded_file',filename=filename))
            if(filename not in files):
                classification_done = False
            return render_template('upload.html',obj={"err":filename+" uploaded successfully"})
    return render_template("upload.html",obj={"err":""})

from flask import send_from_directory

'''@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
    return render_template('upload_successful.html')'''

@app.route('/')
def main():
    return render_template("index.html")
@app.route('/index.html')
def ind():
    return render_template("index.html")

@app.route('/listen.html')
def listen():
    import os
    try:
        filenames=os.listdir("./static/uploads")
        #return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
        if len(filenames)==0:
            raise Exception("FileNotFoundError")
    except:
        return render_template("no_files.html")
    return render_template("listen.html",obj={"filenames":filenames})

@app.route('/classify.html')
def load_classify():
    import os
    try:
        filenames=os.listdir("./static/uploads")
        #return send_from_directory(app.config['UPLOAD_FOLDER'],filename)
        if len(filenames)==0:
            raise Exception("FileNotFoundError")
    except:
        return render_template("no_files.html")
    for x in range(len(filenames)):
        filenames[x]=str(x+1)+") "+filenames[x]
    return render_template("load_classify.html",obj={"filenames":filenames})

@app.route('/load_div')
def func():
    return render_template("loading.html")

@app.route('/classify_result')
def classify():
    global classification_done
    global result
    global files
    if(classification_done):
        print("Loading existing results")
        return render_template("classify.html",obj={"count":len(files),"filenames":files,"genres":result,"err":''})
    features=None
    print(files)
    print(result)
    try:
        extra_files = os.listdir("./static/uploads")
        extra_files = [x for x in extra_files if x not in files]
        files.extend(extra_files)
        if len(files)==0:
            raise FileNotFoundError
        ex = fe(audio_files_dir='./static/uploads/')
        features = ex.extract(extra_files)
        if(CONSIDER_DATASET):
                x = features.shape[0]
                features = np.concatenate((features,feat),axis=0)
                features = preprocessing.scale(features)
                features = features[:x]
                print("Features shape : ",features.shape)
        else:
                features = preprocessing.scale(features)
        result.extend(EnsembleModel.predict(features))
        classification_done = True
        print(result)
        ex.revert_changes()
    except FileNotFoundError:
        return render_template("classify.html",obj={"count":len(files),"filenames":files,"genres":result,"err":"No files to classify"})
    except:
        return render_template("classify.html",obj={"count":len(files),"filenames":files,"genres":result,"err":"Unexpected error"})
    return render_template("classify.html",obj={"count":len(files),"filenames":files,"genres":result,"err":''})


def get_analysis_data():
    global classification_done
    global result
    global files
    print(classification_done)
    analysis_data = {}
    if ( classification_done == True):
        for i in range(len(result)):
            graph = Digraph()
            Top_level = path_dictionary[result[i]][0]
            Second_level = path_dictionary[result[i]][1]
            Final_level = result[i]
            graph.node('A', Top_level)
            graph.node('B', Second_level)
            graph.node('C', Final_level)
            graph.edges(['AB', 'BC'])
            graph.save(TMP_FOLDER+'/'+ files[i]+'.dot')
            subprocess.check_call(['dot', '-Tpng', TMP_FOLDER + '/'+ files[i]+'.dot', '-o', TMP_FOLDER + '/' + files[i]+'.png'])
            analysis_data[files[i]] =  TMP_FOLDER + '/' + files[i]+'.png'
        print(analysis_data)
        return {"analysis" : analysis_data }
    else:
        print("classification_done is False")
        return {"analysis" : "NONE"}

@app.route('/analysis.html')
def analysis():
    global classification_done
    print(classification_done)
    return render_template('analysis.html', obj = get_analysis_data())

# In[ ]:


app.run(threaded=True)
