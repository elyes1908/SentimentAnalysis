import datetime
import json

import joblib
from flask import Flask, render_template, redirect, url_for, session, request, jsonify
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
from flask import Flask, send_file
from pymongo import MongoClient
import pandas as pd
import torch
import numpy as np
from keras.utils import pad_sequences

from transformers import CamembertTokenizer, CamembertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine", use_fast=True)
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

app = Flask(__name__)
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secret'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# client = MongoClient('mongodb://dxcUser:dxc@localhost:27017/?authMechanism=DEFAULT&authSource=dxc&replicaSet=rs1')
client = MongoClient('localhost', 27017)
db1 = client['dxc']

# récupérer les données de la collection "documents"
documents = db1['DXCnew'].find()

# Defining constants
epochs = 15
MAX_LEN = 150
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# Initializer CamemBERT tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base', use_fast=True)
#
# Mymodel = CamembertForSequenceClassification.from_pretrained("camembert-base", num_labels=2)
# Mymodel.to(device)
# joblib.dump(Mymodel, 'model/sentiments1.pt')
Mymodel = joblib.load('model/sentiments1.pt')


# Mymodel.load_state_dict(torch.load("C:\\Users\\PC-ROG\\SA\\SA\\modele\\sentiments1.pt"))


# @app.route('/test', methods=('GET', 'POST'))
# def test():
#     all_doc = db1['testFlask'].find()
#     # all_docs=list(all_doc)
#     data_xls = pd.DataFrame(list(all_doc))
#     del data_xls['_id']
#
#     data_xls = data_xls[:10]
#     data_xls['prediction'] = ''
#     for i in range(data_xls.shape[0]):
#         data_xls['prediction'][i] = nlp(data_xls['CommentValue'][i])[0]['label']
#         print(data_xls['prediction'][i])
#
#     jsonfiles = json.loads(data_xls.to_json(orient='records'))
#     # print(datetime.datetime.now() - now)
#
#     collection = db1['dataAfterPreduction']
#     collection.insert_many(jsonfiles)
#
#     # 3awedt a3meltha mara o5ra 5ater ki n3ayat l jsonfiles tjibli erreur
#     jsonfiles1 = json.loads(data_xls.to_json(orient='records'))
#
#     # return jsonify({'file': jsonfiles1 })
#
#     print("$$$$$$$$$$$$$$$$$$$$")
#     print(data_xls)
#     d = data_xls.to_dict(orient='records')
#
#     return render_template('test.html', todos=d)


@app.route('/testModele', methods=('GET', 'POST'))
def testModele():
    dict = {0: 'Negative', 1: 'Positive'}
    all_doc = db1['testFlask'].find()
    # all_docs=list(all_doc)
    data_xls = pd.DataFrame(list(all_doc))
    del data_xls['_id']

    data_xls = data_xls[:10]
    data_xls['prediction'] = ''

    ### tokozier pour le modéle
    # Encode the comments
    tokenized_comments_ids = [tokenizer.encode(comment, add_special_tokens=True, max_length=MAX_LEN) for comment in
                              data_xls['CommentValue']]
    # Pad the resulted encoded comments
    tokenized_comments_ids = pad_sequences(tokenized_comments_ids, maxlen=MAX_LEN, dtype="long", truncating="post",
                                           padding="post")

    # Create attention masks
    attention_masks = []
    for seq in tokenized_comments_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(tokenized_comments_ids)
    prediction_masks = torch.tensor(attention_masks)

    # Apply the finetuned model (Camembert)
    flat_pred = []
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = Mymodel(prediction_inputs.to(device), token_type_ids=None, attention_mask=prediction_masks.to(device))
        # outputs
        logits = outputs[0]
        print(logits)
        logits = logits.detach().cpu().numpy()
        print("=========")
        print(np.argmax(logits, axis=1).flatten())
        flat_pred.extend(np.argmax(logits, axis=1).flatten())

    for i in range(len(flat_pred)):
        print('Comment: ', data_xls['CommentValue'][i])
        print('Label', dict[flat_pred[i]])
        data_xls['prediction'][i] = dict[flat_pred[i]]
        print(data_xls['prediction'][i])

    jsonfiles = json.loads(data_xls.to_json(orient='records'))
    # print(datetime.datetime.now() - now)
    collection = db1['dataAfterPreduction']
    collection.insert_many(jsonfiles)

    # 3awedt a3meltha mara o5ra 5ater ki n3ayat l jsonfiles tjibli erreur
    jsonfiles1 = json.loads(data_xls.to_json(orient='records'))
    Task = []
    CommentValue = []
    prediction = []
    for file in jsonfiles1:
        Task.append(file['Task'])
        CommentValue.append(file['CommentValue'])
        prediction.append(file['prediction'])

    # return jsonify({'file': jsonfiles1})
    return render_template('test.html', todos={'Task': Task, 'CommentValue': CommentValue, 'prediction': prediction}, length=len(Task))


#########################################################################

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)


class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})

    submit = SubmitField("Register")

    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                "That username already exists. Please choose a different one.")


class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")


@app.route('/')
@login_required
def index():
    return render_template('index.html', session=session)


@app.route('/process', methods=['POST'])
@login_required
def process():
    comment = request.form['comment_input']
    print(comment)
    if comment:
        class_ = nlp(comment)[0]['label']
        score_ = nlp(comment)[0]['score']
        print(f"The sentiment of the text is {class_}")
        return jsonify({'class_': class_, 'score_': round(score_, 2)})

    return jsonify({'error': 'Missing data!'})


@app.route('/telecharger-fichier')
def telecharger_fichier():
    # se connecter à la base de données
    # se connecter à la base de données
    client = MongoClient('mongodb://dxcUser:dxc@localhost:27017/?authMechanism=DEFAULT&authSource=dxc&replicaSet=rs1')
    db = client['dxc']

    # récupérer les données de la collection "documents"
    documents = db['DXCnew'].find()

    # créer un DataFrame pandas à partir des données
    df = pd.DataFrame(documents)

    # générer le fichier Excel
    fichier_excel = df.to_excel('documents.xlsx', index=False)

    # renvoyer le fichier Excel au client
    return send_file('documents.xlsx', as_attachment=True)


@app.route("/upload", methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        now = datetime.datetime.now()

        f = request.files['file']

        data_xls = pd.read_excel(f)
        data_xls = data_xls[:10]
        data_xls['prediction'] = ''
        for i in range(data_xls.shape[0]):
            data_xls['prediction'][i] = nlp(data_xls['CommentValue'][i])[0]['label']

        jsonfiles = json.loads(data_xls.to_json(orient='records'))
        print(datetime.datetime.now() - now)
        return jsonify({'file': jsonfiles, 'len': data_xls.shape[0]})
    else:
        return 'Oops'


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            session['username'] = form.username.data
            return redirect(url_for('index'))
        else:
            return redirect(url_for('login'))
    return render_template('login.html', form=form)


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html', form=form)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run(debug=True, port=5000, reload=True)
