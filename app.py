import datetime
import json

import joblib
import pandas as pd
from flask import Flask, render_template, redirect, url_for, session, request, jsonify
from flask_bcrypt import Bcrypt
from flask_login import UserMixin, LoginManager, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError

app = Flask(__name__)
bcrypt = Bcrypt(app)
db = SQLAlchemy(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'secret'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


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
        sentiment = joblib.load('predict.pkl')
        class_ = sentiment(comment)[0]['label']
        score_ = sentiment(comment)[0]['score']
        print(f"The sentiment of the text is {class_}")
        return jsonify({'class_': class_, 'score_': round(score_, 2)})

    return jsonify({'error': 'Missing data!'})


@app.route("/upload", methods=['GET', 'POST'])
@login_required
def upload_file():
    if request.method == 'POST':
        now = datetime.datetime.now()
        f = request.files['file']
        data_xls = pd.read_excel(f)
        data_xls = data_xls[:4]
        sentiment = joblib.load('predict.pkl')
        data_xls['prediction'] = ''
        for i in range(data_xls.shape[0]):
            data_xls['prediction'][i] = sentiment(data_xls['CommentValue'][i])[0]['label']

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
    app.run(debug=True, port=5000)
