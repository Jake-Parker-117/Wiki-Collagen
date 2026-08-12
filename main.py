#Copyright 2026 Jacob Parker and Jordi Bella

#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at

    #http://www.apache.org/licenses/LICENSE-2.0

#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
from flask import Flask, render_template, request, send_from_directory, redirect, url_for, make_response
from flask_mail import Mail, Message
from Collagens import get_cols, label_cols, cleanup_files
from CollagenAI import CollagenFamilyClassifier, CONFIG, FAMILIES, LABEL2ID, ID2LABEL, load_model, standard_predict, predict_long_sequence, get_seqs, make_prediction
from waitress import serve
from werkzeug.exceptions import HTTPException, BadRequest, NotFound, Forbidden
from dotenv import load_dotenv
import os, uuid, time, shutil, secrets, queue, threading, gc
import torch

load_dotenv()

app = Flask(__name__)
cleanup_files("sessions", 3600)

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/CollagenAI')
def CollagenAI():
    return render_template('CollagenAI.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/citation')
def citation():
    return render_template('citation.html')

user_db = {}

#Route for file inputs which runs all the functions given the inputs and then redirects to a different link to show the files depending on user id
@app.route("/collagens/file_input", methods=["POST"])
def collagens_file():

    #creating random user id and creating a working directory within the sessions folder labeled as the request_id (user id)  
    request_id = str(uuid.uuid4())
    workdir = os.path.join("sessions", request_id)
    os.makedirs(workdir)
    
    owner_token = secrets.token_hex(32)
    user_db[request_id] = owner_token
    
    #requesting input file, creating individual file within current workdir within sessions and the assigned random user id
    cols_file = request.files['fasta_seqs']
    input_fasta = os.path.join(workdir, "input.fasta")
    cols_file.save(input_fasta) #save function effienctly saves the file by loading it into the other file in chuncks of 16Kb
    
    #getting all inputs from web page and creating output files in working directory. Handling input errors for integers:
    try:
        min_col = int(request.form['col_len_min'])
        max_inter = int(request.form['interruption_max'])
    except(KeyError, ValueError): 
        raise BadRequest(description="Incorrect input. Please use positive integers as inputs regarding COL domain length or interuption length.")
    
    col_txt = os.path.join(workdir, "col.txt")
    col_html = os.path.join(workdir, "col_html.html")
    col_table = os.path.join(workdir, "col_table.txt")
    
    get_cols(input_fasta, col_txt, min_col)
    label_cols(col_txt, col_html, col_table, max_inter)
    
    response = make_response(redirect(url_for("results", user_id=request_id)))
    
    response.set_cookie('viewer_device_id', owner_token, httponly=True, samesite='Lax', max_age=3600) # implementing cookie storage for browser lock
    
    return response


#Route for file inputs which runs all the functions given the inputs and then redirects to a different link to show the files depending on user id
@app.route("/collagens/text_input", methods=["POST"])
def collagens_txt():

    #creating random user id and creating a working directory within the sessions folder labeled as the request_id (user id)  
    request_id = str(uuid.uuid4())
    workdir = os.path.join("sessions", request_id)
    os.makedirs(workdir)
    
    owner_token = secrets.token_hex(32)
    user_db[request_id] = owner_token
    
    #requesting input text and validating it
    cols_text_input = request.form['fasta_text']
    input_fasta = os.path.join(workdir, "input.fasta")
    with open(input_fasta, 'w', encoding="utf-8") as f:
        f.write(cols_text_input)

    #getting numerical inputs and handling input errors:
    try:
        min_col = int(request.form['col_len_min'])
        max_inter = int(request.form['interruption_max'])
    except(ValueError): 
        raise BadRequest(description="Incorrect input. Please use positive integers as inputs regarding minimum COL domain length and maximuum interuption length.")
    except(KeyError): 
        raise BadRequest(description="Please provide inputs for minimum COL domain length and maximum interuption length.")
    
    #creating an input.fasta file for counting inputted sequences later and creating all the other files in the working directory
    col_txt = os.path.join(workdir, "col.txt")
    col_html = os.path.join(workdir, "col_html.html")
    col_table = os.path.join(workdir, "col_table.txt")
    
    get_cols(input_fasta, col_txt, min_col)
    label_cols(col_txt, col_html, col_table, max_inter)
    
    response = make_response(redirect(url_for("results", user_id=request_id)))
    
    response.set_cookie('viewer_device_id', owner_token, httponly=True, samesite='Lax', max_age=3600)
    
    return response



#setting up user isolated routes per user id so that the results page url isn't the same for 2 people
@app.route("/results/<user_id>", methods=["GET"])
def results(user_id):
    
    expected_cookie = user_db.get(user_id)
    if not expected_cookie:
        raise NotFound(description="The session you are trying to access has been removed. Please run your query again.") #Check for user data on system
    user_cookie = request.cookies.get('viewer_device_id')
    if user_cookie != expected_cookie: #compare user cookie token to stored token
        raise Forbidden(description="You do not have permission to view these results. Please use the browser used to submit the query.") 
    
    #getting the working directory and files for counting
    workdir = os.path.join("sessions", user_id)
    input_fasta = os.path.join(workdir, "input.fasta")
    col_txt = os.path.join(workdir, "col.txt")
    
    #counting sequences in origninal fasta file and verifying that the file still exists
    try:
        with open(input_fasta, 'r', encoding='utf-8') as yfile:
            seq_num = 0
            for yline in yfile:
                if yline.startswith(">"):
                    seq_num += 1
    except:
        raise NotFound(description="The session you are trying to access has been removed. Please run your query again.")
    
    #counting sequences that are over the 30AA length requirement (ie. are in the col.txt file)
    with open(col_txt, 'r', encoding='utf-8') as xfile:
        col_num = 0
        for line in xfile:
            if line.startswith(">"):
                col_num += 1
    
    return render_template("collagens.html", seqs=seq_num, collagen_num=col_num, user_id=user_id,)


#setting up a route for each user to view and download their files
@app.route("/sessions/<user_id>/<filename>", methods=["GET"])
def sessions(user_id, filename):
    
    expected_cookie = user_db.get(user_id)
    if not expected_cookie:
        raise NotFound(description="The session you are trying to access has been removed. Please run your query again.") #Check for user data on system
    
    user_cookie = request.cookies.get('viewer_device_id')
    if user_cookie != expected_cookie: #compare user cookie token to stored token
        raise Forbidden(description="You do not have permission to view these results. Please use the browser used to submit the query.")
    
    return send_from_directory(os.path.join("sessions", user_id), filename)

# Creating Queue system for AI due to GPU constraints
request_queue = queue.Queue()
results_registry = {}
registry_lock = threading.Lock()

#Global objects for model instance:
model = None
tokenizer = None
device = None

# Creating a worker loop to load model onto GPU and process each fasta file one at a time
def gpu_worker_loop():
    """
    Background GPU worker
    """
    
    global model, tokenizer, device
    
    window_size = CONFIG['window_size']
    model, tokenizer, device = load_model() # loads model on startup
    
    while True:
        task_id, input_fasta, colAI_txt, classification_txt, confidence_threshold = request_queue.get() # pulls all info for request from the functions below
        
        try:
            get_seqs(input_fasta, colAI_txt)
            make_prediction(colAI_txt, classification_txt, confidence_threshold, model, tokenizer, device, window_size)

            with registry_lock:
                results_registry[task_id] = {"status": "success"}

        except Exception as e:
            with registry_lock:
                results_registry[task_id] = {"status": "error", "message": str(e)}
        
        finally:
            if device and device.type == 'cuda':
                gc.collect()
                torch.cuda.empty_cache()
        
        request_queue.task_done()

# Route for CollagenAI file input
@app.route("/CollagenAI/file_input", methods=["POST"])
def CollagenAI_file():

    #creating random user id and creating a working directory within the sessions folder labeled as the request_id (user id)  
    request_id = str(uuid.uuid4())
    workdir = os.path.join("sessions", request_id)
    os.makedirs(workdir)
    
    owner_token = secrets.token_hex(32)
    user_db[request_id] = owner_token
    
    #requesting input file, creating individual file within current workdir within sessions and the assigned random user id
    cols_file = request.files['fasta_seqs']
    input_fasta = os.path.join(workdir, "input.fasta")
    cols_file.save(input_fasta) #save function effienctly saves the file by loading it into the other file in chunks of 16Kb
    
    #getting numerical input and handling input errors:
    try:
        confidence_threshold = int(request.form['confidence_threshold'])
    except(ValueError): 
        raise BadRequest(description="Incorrect input. Please use positive integers as the input for the confidence threshold.")
    except(KeyError): 
        raise BadRequest(description="Please provide input for the minimum confidence threshold.")
    
    colAI_txt = os.path.join(workdir, "colAI.txt")
    classification_txt = os.path.join(workdir, "Classifications.txt")
    
    #Adding current request to queue with all input and output objects
    request_queue.put((request_id, input_fasta, colAI_txt, classification_txt, confidence_threshold))
    # Creating a timeout counter
    timeout = 300
    start_time = time.time()
    
    # Implementing Synchronous waiting pool to stop users from having requests handled simultaneously
    while True:
        with registry_lock:
            if request_id in results_registry:
                result = results_registry.pop(request_id)
                if result["status"] == "success":
                    break
                else:
                    raise BadRequest(description=f"Prediction failed: {result['message']}") # Show error message if prediction failed
        if time.time() - start_time > timeout: # Timeout the request if its taking too long
            raise BadRequest(description="The server queue took too long to process this FASTA file. Please try again or a smaller batch.")
    
        time.sleep(1.5) # Stops CPU from constantly being used to check if its finished yet
    
    response = make_response(redirect(url_for("resultsAI", user_id=request_id)))
    response.set_cookie('viewer_device_id', owner_token, httponly=True, samesite='Lax', max_age=3600) # implementing cookie storage for browser lock
        
    return response

# Route for CollagenAI file input
@app.route("/CollagenAI/text_input", methods=["POST"])
def CollagenAI_text():

    #creating random user id and creating a working directory within the sessions folder labeled as the request_id (user id)  
    request_id = str(uuid.uuid4())
    workdir = os.path.join("sessions", request_id)
    os.makedirs(workdir)
    
    owner_token = secrets.token_hex(32)
    user_db[request_id] = owner_token
    
    #requesting input file, creating individual file within current workdir within sessions and the assigned random user id
    
    #requesting input text and validating it
    cols_text_input = request.form['fasta_text']
    input_fasta = os.path.join(workdir, "input.fasta")
    with open(input_fasta, 'w', encoding="utf-8") as f:
        f.write(cols_text_input)
        
    #getting numerical input and handling input errors:
    try:
        confidence_threshold = int(request.form['confidence_threshold'])
    except(ValueError): 
        raise BadRequest(description="Incorrect input. Please use positive integers as the input for the confidence threshold.")
    except(KeyError): 
        raise BadRequest(description="Please provide input for the minimum confidence threshold.")
    
    colAI_txt = os.path.join(workdir, "colAI.txt")
    classification_txt = os.path.join(workdir, "Classifications.txt")
    
    #Adding current request to queue with all input and output objects
    request_queue.put((request_id, input_fasta, colAI_txt, classification_txt, confidence_threshold))
    # Creating a timeout counter
    timeout = 300
    start_time = time.time()
    
    # Implementing Synchronous waiting pool to stop users from having requests handled simultaneously
    while True:
        with registry_lock:
            if request_id in results_registry:
                result = results_registry.pop(request_id)
                if result["status"] == "success":
                    break
                else:
                    raise BadRequest(description=f"Prediction failed: {result['message']}") # Show error message if prediction failed
        if time.time() - start_time > timeout: # Timeout the request if its taking too long
            raise BadRequest(description="The server queue took too long to process this FASTA file. Please try again or a smaller batch.")
        
        time.sleep(1.5) # Stops CPU from constantly being used to check if its finished yet
    
    response = make_response(redirect(url_for("resultsAI", user_id=request_id)))
    response.set_cookie('viewer_device_id', owner_token, httponly=True, samesite='Lax', max_age=3600) # implementing cookie storage for browser lock
        
    return response

@app.route("/resultsAI/<user_id>", methods=["GET"])
def resultsAI(user_id):
    
    expected_cookie = user_db.get(user_id)
    if not expected_cookie:
        raise NotFound(description="The session you are trying to access has been removed. Please run your query again.") #Check for user data on system
    user_cookie = request.cookies.get('viewer_device_id')
    if user_cookie != expected_cookie: #compare user cookie token to stored token
        raise Forbidden(description="You do not have permission to view these results. Please use the browser used to submit the query.") 
    
    #getting the working directory and files for counting
    workdir = os.path.join("sessions", user_id)
    colAI_txt = os.path.join(workdir, "Classifications.txt")
    
    #counting sequences in origninal fasta file and verifying that the file still exists
    try:
        with open(colAI_txt, 'r', encoding='utf-8') as f:
            seq_num = 0
            for line in f:
                if line.startswith(">"):
                    seq_num += 1
    except:
            raise NotFound(description="The session you are trying to access has been removed. Please run your query again.")
    
    return render_template("CollagenAI_results.html", user_id=user_id, seqs=seq_num)

#Error handling
@app.errorhandler(HTTPException)
def handle_exeption(e):
    return render_template("errors.html",
                            e_title=e.name,
                            e_msg=e.description,
                            ), e.code #returns the error code to terminal aswell as rendering to template

#Configuring email sending system thru env
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')
app.config['MAIL_PORT'] = os.getenv('MAIL_PORT')
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
#creating mail class after configuration of Mail
mail = Mail(app)

@app.route('/message_sent')
def message_sent():
    return render_template('message_sent.html')

#Creating mail contact form usability
@app.route('/contact', methods=['GET','POST'])
def contact():
    if request.method == 'POST':
        #requesting details from the contact form
        name = request.form.get('contact_name')
        email = request.form.get('contact_email')
        subject = request.form.get('contact_subject')
        message = request.form.get('contact_message')
        recipient = os.getenv('RECIPIENT')

        #creating message
        msg = Message(
            subject=f"Wiki Collagens contact form submission",
            recipients=[recipient],
            body=f"""
            New message from contact form:

            Name: {name}
            Subject: {subject}
            email: {email}

            {message}
            """,
            reply_to=email
        )

        #sending email and debugging if not:
        try: 
            mail.send(msg)
            return redirect(url_for('message_sent'))
        except Exception as e:
            raise e
    return render_template('contact.html')



if __name__ == '__main__':
    # Launching worker thread:
    worker_thread = threading.Thread(target=gpu_worker_loop, daemon=True)
    worker_thread.start()
    
    # Run Flask development server
    app.run(debug=True, host='0.0.0.0', port=5000)

    # Run Flask production server with Waitress
    #serve(app, host='0.0.0.0', port=5000, threads=9, max_request_body_size=1073741824)