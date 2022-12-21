from flask import Flask, request
from credit_card.exception import CreditCardException
from credit_card.logger import logging, get_log_dataframe
import os, sys
from credit_card.config.configuration import Configuration
from flask import send_file, abort, render_template
from credit_card.pipeline.pipeline import Pipeline
from credit_card.constant import CONFIG_DIR, get_current_time_stamp
from credit_card.entity.credit_card_predictor import Credit_Card_Predictor, CreditCardData
import json
from credit_card.util.util import read_yaml, write_yaml

ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = 'credit_card_logs'
PIPELINE_FOLDER_NAME = 'credit_card'
SAVED_MODELS_DIR_NAME = 'saved_models'
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, 'model.yaml')
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

CREDIT_CARD_DATA_KEY = "credit_card_data"
DEFAULT_PAYMENT_NEXT_MONTH_KEY = "default_payment_next_month"

app = Flask(__name__)

@app.route('/artifact', defaults={'req_path': 'credit_card'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    os.makedirs("credit_card", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)
     
@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiment_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)

@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuration(time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiment_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        CREDIT_CARD_DATA_KEY: None,
        DEFAULT_PAYMENT_NEXT_MONTH_KEY: None
    }

    if request.method == 'POST':
        LIMIT_BAL = int(request.form['LIMIT_BAL'])
        SEX = request.form['SEX']
        EDUCATION = request.form['EDUCATION']
        MARRIAGE = request.form['MARRIAGE']
        AGE = int(request.form['AGE'])
        PAY_0 = request.form['PAY_0']
        PAY_2 = request.form['PAY_2']
        PAY_3 = request.form['PAY_3']
        PAY_4 = request.form['PAY_4']
        PAY_5 = request.form['PAY_5']
        PAY_6 = request.form['PAY_6']
        BILL_AMT1 = int(request.form['BILL_AMT1'])
        BILL_AMT2 = int(request.form['BILL_AMT2'])
        BILL_AMT3 = int(request.form['BILL_AMT3'])
        BILL_AMT4 = int(request.form['BILL_AMT4'])
        BILL_AMT5 = int(request.form['BILL_AMT5'])
        BILL_AMT6 = int(request.form['BILL_AMT6'])
        PAY_AMT1 = int(request.form['PAY_AMT1'])
        PAY_AMT2 = int(request.form['PAY_AMT2'])
        PAY_AMT3 = int(request.form['PAY_AMT3'])
        PAY_AMT4 = int(request.form['PAY_AMT4'])
        PAY_AMT5 = int(request.form['PAY_AMT5'])
        PAY_AMT6 = int(request.form['PAY_AMT6'])

        credit_card_data = CreditCardData(
            LIMIT_BAL=LIMIT_BAL,
            SEX=SEX,
            EDUCATION=EDUCATION,
            MARRIAGE=MARRIAGE,
            AGE=AGE,
            PAY_0=PAY_0,
            PAY_2 = PAY_2,
            PAY_3=PAY_3,
            PAY_4=PAY_4,
            PAY_5=PAY_5,
            PAY_6=PAY_6,
            BILL_AMT1=BILL_AMT1,
            BILL_AMT2=BILL_AMT2,
            BILL_AMT3=BILL_AMT3,
            BILL_AMT4=BILL_AMT4,
            BILL_AMT5=BILL_AMT5,
            BILL_AMT6=BILL_AMT6,
            PAY_AMT1=PAY_AMT1,
            PAY_AMT2=PAY_AMT2,
            PAY_AMT3=PAY_AMT3,
            PAY_AMT4=PAY_AMT4,
            PAY_AMT5=PAY_AMT5,
            PAY_AMT6=PAY_AMT6)
        credit_card_df = credit_card_data.credit_card_input_data_frame()
        credit_card_predictor = Credit_Card_Predictor(model_dir=MODEL_DIR)
        default_payment_next_month = credit_card_predictor.predict(X=credit_card_df)
        context = {
            CREDIT_CARD_DATA_KEY: credit_card_data.get_credit_card_data_as_dict(),
            DEFAULT_PAYMENT_NEXT_MONTH_KEY: default_payment_next_month,
        }
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)
    
@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0")