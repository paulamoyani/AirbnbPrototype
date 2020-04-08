# Gevent needed for sockets
from gevent import monkey
monkey.patch_all(thread=False)

# Imports
import os
from flask import Flask, render_template, request, json
# from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
# import filters
import time
import pickle
import numpy as np
import json
import pickle
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
import math
import csv


import numpy as np
import pandas as pd
import re
from nltk.tokenize import TreebankWordTokenizer

import matplotlib
import sklearn
from IPython.core.display import display, HTML

""
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.9)

# TO WORK WITH
import pandas as pd
import numpy as np
from numpy import set_printoptions

# HIDE WARNINGS
import warnings
warnings.filterwarnings('ignore')

# PREPROCESSING & MODEL SELECTION
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, LassoCV, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import SCORERS
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm

# print(SCORERS.keys())

# PLOTTING
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook")
# get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from sklearn import tree
from graphviz import Source
from matplotlib.pylab import rcParams
import matplotlib.lines as mlines
from yellowbrick.classifier import PrecisionRecallCurve
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
import plotly.express as px
import scipy.cluster.hierarchy as sch
from sklearn.metrics import classification_report


# STANDARD MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# ENSEMBLE
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# XGBOOST
from xgboost import XGBClassifier
from xgboost import XGBRegressor
import xgboost as xgb

# CLUSTERING
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

# PICKLE
from pickle import dump
from pickle import load


start_time = time.time()
# Configure app
socketio = SocketIO()
app = Flask(__name__)
app.config.from_object(os.environ['APP_SETTINGS'])


# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

# DB
# db = SQLAlchemy(app)

# Import + Register Blueprints
# from app.accounts import accounts as accounts
# app.register_blueprint(accounts)
# from app.irsystem import irsystem as irsystem
# app.register_blueprint(irsystem)

# Initialize app w/SocketIO
socketio.init_app(app)

# HTTP error handling
@app.errorhandler(404)
def not_found(error):
    return render_template("404.html"), 404

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/filters', methods=['GET'])
def filters():

    def str_to_bool(s):
        if s == 'true':
             return True
        elif s == 'false':
             return False

    # text inputs
    host_response_rate = float(request.args.get("host_response_rate"));
    date = request.args.get("date");
    security_deposit = float(request.args.get("security_deposit"));
    cleaning_fee = float(request.args.get("cleaning_fee"));
    # standardizing cleaning, security fee, and host_response_rate
    data = pd.read_csv("Data/listings_8.csv")
    scaler = MinMaxScaler(feature_range=(0,1))

    sec = data[["security_deposit"]].copy()
    df2 = pd.DataFrame([[security_deposit]], columns=['security_deposit'])
    sec = sec.append(df2)
    sec = scaler.fit_transform(sec)
    security_deposit = float(sec[-1:])

    clean = data[["cleaning_fee"]].copy()
    df2   = pd.DataFrame([[cleaning_fee]], columns=['cleaning_fee'])
    clean = clean.append(df2)
    clean = scaler.fit_transform(clean)
    cleaning_fee = float(clean[-1:])

    hrr = data[["host_response_rate"]].copy()
    df2 = pd.DataFrame([[host_response_rate]], columns=['host_response_rate'])
    hrr = hrr.append(df2)
    hrr = scaler.fit_transform(hrr)
    host_response_rate = float(hrr[-1:])

    # binary inputs
    description = str_to_bool(request.args.get("description"));
    transit_bool = str_to_bool(request.args.get("transit_bool"));
    host_about_bool = str_to_bool(request.args.get("host_about_bool"));
    nopets_bool = str_to_bool(request.args.get("nopets_bool"));
    nosmok_bool = str_to_bool(request.args.get("nosmok_bool"));
    host_identity_verified = str_to_bool(request.args.get("host_identity_verified"));
    is_location_exact = str_to_bool(request.args.get("is_location_exact"));
    Wheelchair_bool = str_to_bool(request.args.get("Wheelchair_bool"));
    TV_bool = str_to_bool(request.args.get("TV_bool"));
    Hair_Dryer_bool = str_to_bool(request.args.get("Hair_Dryer_bool"));
    twentyfour_Hour_Check_in_bool = str_to_bool(request.args.get("twentyfour_Hour_Check_in_bool"));
    Doorman_bool = str_to_bool(request.args.get("Doorman_bool"));
    Kitchen_bool = str_to_bool(request.args.get("Kitchen_bool"));
    Smoke_Detector_bool = str_to_bool(request.args.get("Smoke_Detector_bool"));
    Heating_bool = str_to_bool(request.args.get("Heating_bool"));
    Clothes_Dryer_bool = str_to_bool(request.args.get("Clothes_Dryer_bool"));
    Pets_live_in_flat_bool = str_to_bool(request.args.get("Pets_live_in_flat_bool"));
    Free_Parking_bool = str_to_bool(request.args.get("Free_Parking_bool"));
    Internet_bool = str_to_bool(request.args.get("Internet_bool"));
    AC_bool = str_to_bool(request.args.get("AC_bool"));
    instant_bookable = str_to_bool(request.args.get("instant_bookable"));
    require_guest_profile_picture = str_to_bool(request.args.get("require_guest_profile_picture"));
    require_guest_phone_verification = str_to_bool(request.args.get("require_guest_phone_verification"));
    extra_price_75 = str_to_bool(request.args.get("extra_price_75"));

    # dropdown inputs
    neighborhood = request.args.get("neighborhood");
    property_type = request.args.get("property_type");
    room_type = request.args.get("room_type");
    accomodates = request.args.get("accomodates");
    bedrooms = request.args.get("bedrooms");
    bathrooms = request.args.get("bathrooms");
    extra_guests = request.args.get("extra_guests");
    beds = request.args.get("beds");
    response_time = request.args.get("response_time");
    cancellations = request.args.get("cancellations");


    def recommendations(coefs, airbnb, X_wnei, data, recom):
        '''
        Makes recommendations based on the specific Airbnb's characteristics and the important features recognized by
        the best model in best_models()

        '''

        X_wnei_cols = X_wnei.columns

        for_improvement = []
        for col in range(1,len(X_wnei.columns)):
            if coefs[col] > 0 :
                if airbnb[0][col] < data.iloc[:,col].mean():
                    if "cancellations" not in recom.iloc[0, col] and ("You should be more moderate on cancellations" not in for_improvement and "You should be less strict on cancellations" not in for_improvement):
                        if "messages" not in recom.iloc[0, col] and ("You should answer messages sooner" not in for_improvement):
                            sent = recom.iloc[0, col]
                            for_improvement.append(sent)
            elif coefs[col] < 0:
                if airbnb[0][col] > data.iloc[:,col].mean():
                    if "cancellations" not in recom.iloc[1, col] and ("You should be more moderate on cancellations" not in for_improvement and "You should be less strict on cancellations" not in for_improvement):
                        if "messages" not in recom.iloc[1, col] and ("You should answer messages sooner" not in for_improvement):
                            sent = recom.iloc[1, col]
                            for_improvement.append(sent)

    #     print(for_improvement)
        # If there are improvements to make...
        if len(for_improvement) != 0:
        #   Concatenating messages
            mess1 = "In order for your Airbnb to be truly competitive, make sure you do the following things: \n"
            mess2 = "\n -- ".join(for_improvement)
            message = mess1 + "\n" + mess2

            return message
        # If there are NO improvements to make...
        else:
            return "Your Airbnb is very competitive compared to all the other Airbnbs in Boston!"




    def best_models(data):
        '''
        Implements Ridge, Lasso and ElasticNet to determine best model, based on inputted data.

        '''
        MSEs = []


        data=data.dropna()

        data=data[data["number_of_reviews"]>=1]
        data=data[data["guests_included"]>=1]

        data['price_per_person'] = data['price_per_night']/(data['guests_included']+data['extra_people'])

        data['output'] = (data['number_of_reviews']*data['review_scores_rating'])/data['availability_365']

        data = data.drop(["property_type","room_type","description","house_rules", "amenities", "id",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication", "review_scores_value", "number_of_reviews",
                          "reviews_per_month", "availability_365", "neighbourhood_cleansed","bathrooms",
                          "host_is_superhost", "accommodates", "bedrooms","beds", "guests_included","price_per_night",
                          "extra_people", "host_is_superhost", "Kitchen_boolean", "Gym_bool", "Elevator_in_building_bool",
                          "Clothes_Washer_bool", "Internet_bool", "price_per_person"],axis=1)

        scaler=MinMaxScaler(feature_range=(0,1))


        data[['security_deposit']] = scaler.fit_transform(data[['security_deposit']])
        data[['cleaning_fee']] = scaler.fit_transform(data[['cleaning_fee']])
        data[['host_response_rate']] = scaler.fit_transform(data[['host_response_rate']])
    #     data[['price_per_person']] = scaler.fit_transform(data[['price_per_person']])

        data.to_numpy()
        data=pd.DataFrame(data)

        data=data.dropna()

        data['output'][data['output'] > 260] = 260

        y = data["output"] # Target variable (price)
        X_wnei = data.drop(["output"],axis=1)

        # Creating new DF without neighborhood names
        X_wnei.to_csv("X_wnei.csv", index=False)
        data.to_csv("data.csv", index=False)

        #########################################################################################################
        # Ridge Regression

        kfold=KFold(n_splits=10, random_state=7)

        model=Ridge()
        scoring = "neg_mean_squared_error"

        results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
        clf = model.fit(X_wnei, y)
        MSEs.append(("Ridge Regression", results.mean(), clf.coef_))


        #########################################################################################################
        # Lasso Regression

        kfold=KFold(n_splits=10, random_state=7)

        model=Lasso()
        scoring = "neg_mean_squared_error"

        results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
        clf = model.fit(X_wnei, y)
        MSEs.append(("Lasso Regression", results.mean(), clf.coef_))

        #########################################################################################################
        # Elastic Net Regression

        kfold=KFold(n_splits=10, random_state=7)

        model=ElasticNet()
        scoring = "neg_mean_squared_error"

        results=cross_val_score(model, X_wnei, y, cv=kfold, scoring=scoring)
        clf = model.fit(X_wnei, y)
        MSEs.append(("ElasticNet Regression", results.mean(), clf.coef_))
    #     print(MSEs)

        #########################################################################################################

        return (min(MSEs, key = lambda t: t[1])[0], X_wnei, y, min(MSEs, key = lambda t: t[1])[2], data)



    #########################################################################################################
    # #############    MAIN

    data = pd.read_csv("Data/listings_8.csv")
    recom = pd.read_csv("Data/recom.csv")
    best_model = best_models(data)

    method = best_model[0]
    X_wnei = best_model[1]
    y = best_model[2]
    coefs = best_model[3].reshape(33,1)
    data = best_model[4]

    airbnb = pd.read_csv("X_wnei.csv").iloc[2].values.reshape(1,-1)

    a = {  "transit_bool": [transit_bool],
            "nopets_bool":[nopets_bool],
            "nosmok_bool":[nosmok_bool],
            "host_about_bool":[host_about_bool]}

    df = pd.DataFrame(a)

    if response_time=="1Hour":
        df["fewdays_response_time"] = [False]
        df["1day_response_time"] = [False]
        df["1hour_response_time"] = [True]
        df["fewhours_response_time"] = [False]
    elif response_time=="Hours":
        df["fewdays_response_time"] = [False]
        df["1day_response_time"] = [False]
        df["1hour_response_time"] = [False]
        df["fewhours_response_time"] = [True]
    elif response_time=="1Day":
        df["fewdays_response_time"] = [False]
        df["1day_response_time"] = [True]
        df["1hour_response_time"] = [False]
        df["fewhours_response_time"] = [False]
    elif response_time=="Days":
        df["fewdays_response_time"] = [True]
        df["1day_response_time"] = [False]
        df["1hour_response_time"] = [False]
        df["fewhours_response_time"] = [False]

    df["host_response_rate"] = [host_response_rate]
    df["host_identity_verified"] = [host_identity_verified]
    df["is_location_exact"] = [is_location_exact]
    df["Wheelchair_bool"] = [Wheelchair_bool]
    df["TV_bool"] = [TV_bool]
    df["Hair Dryer_bool"] = [Hair_Dryer_bool]
    df["24-Hour_Check-in_bool"] = [twentyfour_Hour_Check_in_bool]
    df["Doorman_bool"] = [Doorman_bool]
    df["Kitchen_bool"] = [Kitchen_bool]
    df["Smoke_Detector_bool"] = [Smoke_Detector_bool]
    df["Clothes_Dryer_bool"] = [Clothes_Dryer_bool]
    df["Pets_live_in_flat_bool"] = [Pets_live_in_flat_bool]
    df["Free_Parking_bool"] = [Free_Parking_bool]
    df["Heating_bool"] = [Heating_bool]
    df["Wireless_Internet_bool"] = [Internet_bool]
    df["AC_bool"] = [AC_bool]
    df["security_deposit"] = [security_deposit]
    df["cleaning_fee"] = [cleaning_fee]
    df["instant_bookable"] = [instant_bookable]

    if cancellations=="SuperStrict":
        df["super_strict_canc"] = [True]
        df["moderate_cancellation"] = [False]
        df["strict_cancellation"] = [False]
        df["flexible_cancellation"] = [False]
    elif cancellations=="Moderate":
        df["super_strict_canc"] = [False]
        df["moderate_cancellation"] = [True]
        df["strict_cancellation"] = [False]
        df["flexible_cancellation"] = [False]
    elif cancellations=="Strict":
        df["super_strict_canc"] = [False]
        df["moderate_cancellation"] = [False]
        df["strict_cancellation"] = [True]
        df["flexible_cancellation"] = [False]
    elif cancellations=="Flexible":
        df["super_strict_canc"] = [False]
        df["moderate_cancellation"] = [False]
        df["strict_cancellation"] = [False]
        df["flexible_cancellation"] = [True]

    df["require_guest_profile_picture"] = [require_guest_profile_picture]
    df["require_guest_phone_verification"] = [require_guest_phone_verification]

    df = df.to_numpy()
    airbnb = df

    # [print(type(df))]
    # [print(type(airbnb))]
    # [print(df.shape)]
    # [print(airbnb.shape)]
    # print(df)
    # print(airbnb)

    # print(recommendations(coefs, df, X_wnei, data, recom))

    improvements = recommendations(coefs, airbnb, X_wnei, data, recom)
    print(improvements)
    return improvements



end_time = time.time()
time_elapsed = end_time - start_time
print("Time Elapsed:", time_elapsed, "seconds")
