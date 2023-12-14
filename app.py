from flask import Flask, render_template, request, redirect, Response, send_from_directory, send_file, url_for
from werkzeug.utils import secure_filename
import io
import os
import time
import glob
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
import cv2
import pytesseract
import math
import re
from PIL import Image
import openpyxl
import shutil
from celery import Celery, Task
import redis
import boto3
import botocore.session

from celery.result import AsyncResult
from time import sleep
app = Flask(__name__)
UPLOAD_FOLDER = './Input/'
PNG_PATH='./PNG/'
OUTPUT_PATH='./OCTess.xlsx'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'zip'}
#shared_path = os.environ['SHARED_PATH']



#pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\\tesseract.exe'

r = redis.from_url(os.environ.get("REDIS_URL"))
celery = Celery(app.import_name, broker='secret', backend=r, s3_access_key_id = 'secret', s3_secret_access_key = 'secret', s3_bucket = 'secret')

celery.conf.update(task_track_started=True, task_time_limit=1000, broker_url=os.environ['REDIS_URL'],result_backend=os.environ['REDIS_URL'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    aws_access_key_id='secret',
    aws_secret_access_key='secret'
)

clientsession = botocore.session.Session()
clientsession.set_credentials(
    access_key='secret',
    secret_key='secret',
    token=None,
)
s3_client = clientsession.create_client('s3')
celery.conf.broker_transport_options = {'s3': s3_client}
## nessesary functions 

def delete_files_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        

def process_Img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    return invert

def get_Object(image, x_coords, y_coords, process=True, lang='eng', config='--psm 7 --oem 1 --tessdata-dir ./tessdata/ -c page_separator='''):
    # x1 --> x2 read left to right; y1 --> y2 read top to bottom
    ROI_dim = image[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
    try:
        if process: ROI_dim = process_Img(ROI_dim)
        #cv2.imshow('TEST', ROI_dim); cv2.waitKey()
        #pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\\tesseract.exe'
        ROI = pytesseract.image_to_string(ROI_dim, lang=lang, config=config)
        object = ROI.replace('\n', '')
    except:
        object = ''
    return object

def get_Object_num(image, x_coords, y_coords, process=True, lang='eng', config='--psm 7 --oem 1 --tessdata-dir ./tessdata/ -c tessedit_char_whitelist=0123456789., --user-patterns ./patterns/gen_user_patterns -c page_separator='''):
    # x1 --> x2 read left to right; y1 --> y2 read top to bottom
    ROI_dim = image[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
    try:
        #cv2.imshow('TEST', ROI_dim); cv2.waitKey()
        if process: ROI_dim = process_Img(ROI_dim)
        #cv2.imshow('TEST', ROI_dim); cv2.waitKey()
        #pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\\tesseract.exe'
        ROI = pytesseract.image_to_string(ROI_dim, lang=lang, config=config)
        object = ROI.replace('\n', '')
    except:
        object = ''
    return object
    

def get_ScanType(image_path, lang='eng', config='--psm 3 --oem 1 --tessdata-dir ./tessdata/ -c preserve_interword_spaces=1 -c page_separator='''):
    image = cv2.imread(image_path)
    #pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\\tesseract.exe'
    scan_type = get_Object(image, x_coords=(2105,2660), y_coords=(270,365), process=False, lang=lang, config=config)
    return scan_type

    
def get_monocular_eye(image_path):
    # Detect which eye is 'blue': define lower and upper limit values for the colour: 'blue'
    lower_blue = np.array([135, 0, 0], dtype = "uint8")
    upper_blue = np.array([145, 0, 0], dtype = "uint8")
    image = cv2.imread(image_path)
    OD_image = image[815:915, 3395:3540]
    mask = cv2.inRange(OD_image, lower_blue, upper_blue)
    is_blue = np.sum(mask)
    if is_blue > 0:
        return 'OD'
    OS_image = image[815:915, 3590:3720]
    mask = cv2.inRange(OS_image, lower_blue, upper_blue)
    is_blue = np.sum(mask)
    if is_blue > 0:
        return 'OS'
    return 'None'


def get_Volume_Object(image, x_coords, y_coords, lang='eng', config='--psm 7 --oem 1 --tessdata-dir ./tessdata/ -c tessedit_char_whitelist=0123456789., --user-patterns ./patterns/vol_user_patterns -c page_separator='''):
    ROI_dim = image[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
    blur = cv2.GaussianBlur(ROI_dim, (1,1), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,7)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ROI = pytesseract.image_to_string(thresh, config=config)
    object = ROI.replace('\n', '')
    return object
    
    
def get_Thickness_Object(image, x_coords, y_coords, lang='eng', config='--psm 7 --oem 1 --tessdata-dir ./tessdata/ -c tessedit_char_whitelist=0123456789., --user-patterns ./patterns/thick_user_patterns -c page_separator='''):
    ROI_dim = image[y_coords[0]:y_coords[1], x_coords[0]:x_coords[1]]
    blur = cv2.GaussianBlur(ROI_dim, (3,3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,7)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ROI = pytesseract.image_to_string(thresh, config=config)
    object = ROI.replace('\n', '')
    return object
    

def find_center(image, param_1=100, param_2=30, min_radius=40, max_radius=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                            param1=param_1, param2=param_2,
                            minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        if circles.shape[1]==1:
            circle = np.uint16(np.around(circles))
            center = (circle[0,0][0], circle[0,0][1])
            return center
        else:
            print('Warning: >1 Central Circles Detected!')
            return False
    else:
        print('Warning: No Circles Detected!')
        return False

# If read text is not a number, return something else. For now, not doing anything.
# In production we can return something like "Manual Verification" so they can easily filter for them and edit them manually
def number_filter(s, error_s=''):
    #new_s = re.sub("[^\d\.]", "", s)
    try:
        float(s)
        return s
    except ValueError:
        return error_s

                    
def extract_Monocular_Report(image_path, mono_eye, show_img=False, verbose=False):
    report_list = []
    image = cv2.imread(image_path)
    name = get_Object(image, x_coords=(850,2000), y_coords=(277,355))
    birthdate = get_Object(image, x_coords=(860,1175), y_coords=(510,585)).replace(" ", "")
    gender = get_Object(image, x_coords=(860,1115), y_coords=(595,670))
    exam_date = get_Object(image, x_coords=(2100,2425), y_coords=(407,495)).replace(" ", "")
    exam_time = get_Object(image, x_coords=(2100,2425), y_coords=(500,580))
    signal_strength = get_Object(image, x_coords=(2100,2275), y_coords=(685,765))
    signal_strength = (signal_strength.replace('/10', '')).strip()
    fovea = get_Object(image, x_coords=(3270,3775), y_coords=(1845,1910))
    fovea = (fovea.replace('Fovea:', '')).strip()
    #circle_image = image[y:y,x:x] # original, tight bounding box
    circle_image = image[1010:1875, 2260:3105] # new, looser bounding box (to prevent big offset crop errors)
    center_x, center_y = find_center(circle_image)
    # Convert totally black portions of the image (black bounding box) to white
    black=np.where((circle_image[:,:,0]==0) & (circle_image[:,:,1]==0) & (circle_image[:,:,2]==0))
    circle_image[black]=(255,255,255)
    # Delete any remaining "ILM-RPE Thickness (um)" text that remains
    circle_image[(1850-1010):(1875-1010), 0:(3105-2260), 0:2] = 255
    
    superior = number_filter(get_Object_num(circle_image, x_coords=(center_x-60,center_x+60), y_coords=(center_y-337,center_y-271)))
    central_superior = number_filter(get_Object_num(circle_image, x_coords=(center_x-60,center_x+60), y_coords=(center_y-174,center_y-111)))
    central = number_filter(get_Object_num(circle_image, x_coords=(center_x-60,center_x+60), y_coords=(center_y-35,center_y+35)))
    central_inferior = number_filter(get_Object_num(circle_image, x_coords=(center_x-60,center_x+60), y_coords=(center_y+110,center_y+174)))
    inferior = number_filter(get_Object_num(circle_image, x_coords=(center_x-60,center_x+60), y_coords=(center_y+270,center_y+336)))
    if (mono_eye == 'OD'):
        nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x+245,center_x+360), y_coords=(center_y-35,center_y+35)))
        central_nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x+90,center_x+190), y_coords=(center_y-35,center_y+35)))
        temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x-369,center_x-255), y_coords=(center_y-35,center_y+35)))
        central_temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x-188,center_x-95), y_coords=(center_y-35,center_y+35)))
    else:
        temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x+245,center_x+360), y_coords=(center_y-35,center_y+35)))
        central_temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x+90,center_x+190), y_coords=(center_y-35,center_y+35)))
        nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x-369,center_x-255), y_coords=(center_y-35,center_y+35)))
        central_nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x-188,center_x-95), y_coords=(center_y-35,center_y+35)))
    volume = number_filter(get_Object_num(image, x_coords=(3260,3445), y_coords=(4655,4740)))
    avg_thickness = number_filter(get_Object_num(image, x_coords=(3660,3855), y_coords=(4655,4740)))

    report_row = {
    'Filename' : os.path.basename(image_path),
    'Name' : name,
    'Birthdate' : birthdate,
    'Gender' : gender,
    'Exam_Date' : exam_date,
    'Exam_Time' : exam_time,
    'Eye' : mono_eye,
    'Signal_Strength' : signal_strength,
    'Superior' : superior,
    'Central_Superior' : central_superior,
    'Nasal' : nasal,
    'Central_Nasal' : central_nasal,
    'Inferior' : inferior,
    'Central_Inferior' : central_inferior,
    'Temporal' : temporal,
    'Central_Temporal' : central_temporal,
    'Central' : central,
    'Volume' : volume,
    'Avg_Thickness' : avg_thickness,
    'Fovea' : fovea
    }
    #print(report_row)
    #cv2.imshow('test', circle_image); cv2.waitKey(0)
    report_list.append(report_row)
    #if show_img:
    #    cv2.imshow(filename, image); cv2.waitKey()
    return report_list
    
def extract_Binocular_Report(image_path, eye, one_eye_recorded=False, show_img=False, verbose=False):
    report_list = []
    image = cv2.imread(image_path)
    name = get_Object(image, x_coords=(850,2000), y_coords=(277,355))
    birthdate = get_Object(image, x_coords=(860,1175), y_coords=(510,585))
    gender = get_Object(image, x_coords=(860,1115), y_coords=(595,670))
    if (eye == 'OD' or one_eye_recorded):
        exam_date = get_Object(image, x_coords=(2100,2425), y_coords=(407,495))
        exam_time = get_Object(image, x_coords=(2100,2425), y_coords=(500,580))
        signal_strength = get_Object(image, x_coords=(2100,2275), y_coords=(685,765))
        signal_strength = (signal_strength.replace('/10', '')).strip()
    else:
        exam_date = get_Object(image, x_coords=(2475,2800), y_coords=(407,495))
        exam_time = get_Object(image, x_coords=(2475,2800), y_coords=(500,580))
        signal_strength = get_Object(image, x_coords=(2475,2800), y_coords=(685,765))
        signal_strength = (signal_strength.replace('/10', '')).strip()
        
    if (eye == 'OD'):
        fovea = get_Object(image, x_coords=(995,1500), y_coords=(2122,2195))
        fovea = (fovea.replace('Fovea:', '')).strip()
        #circle_image = image[2434:3072, 1433:2072] # original, tight bounding box
        circle_image = image[2415:3095, 1410:2100] # new, looser bounding box (to prevent big offset crop errors)
    else:
        fovea = get_Object(image, x_coords=(2990,3520), y_coords=(2122,2195))
        fovea = (fovea.replace('Fovea:', '')).strip()
        #circle_image = image[2434:3072,2481:3120] # original, tight bounding box
        circle_image = image[2415:3095, 2465:3140] # new, looser bounding box (to prevent big offset crop errors)
    
    center_x, center_y = find_center(circle_image)
    
    superior = number_filter(get_Object_num(circle_image, x_coords=(center_x-45,center_x+45), y_coords=(center_y-280,center_y-220)))
    central_superior = number_filter(get_Object_num(circle_image, x_coords=(center_x-45,center_x+45), y_coords=(center_y-140,center_y-80)))
    central = number_filter(get_Object_num(circle_image, x_coords=(center_x-45,center_x+45), y_coords=(center_y-35,center_y+35)))
    central_inferior = number_filter(get_Object_num(circle_image, x_coords=(center_x-45,center_x+45), y_coords=(center_y+80,center_y+140)))
    inferior = number_filter(get_Object_num(circle_image, x_coords=(center_x-45,center_x+45), y_coords=(center_y+220,center_y+280)))
    if (eye == 'OD'):
        nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x+200,center_x+290), y_coords=(center_y-35,center_y+35)))
        central_nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x+70,center_x+150), y_coords=(center_y-35,center_y+35)))
        temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x-290,center_x-205), y_coords=(center_y-35,center_y+35)))
        central_temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x-155,center_x-75), y_coords=(center_y-35,center_y+35)))
    else:
        temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x+200,center_x+290), y_coords=(center_y-35,center_y+35)))
        central_temporal = number_filter(get_Object_num(circle_image, x_coords=(center_x+70,center_x+150), y_coords=(center_y-35,center_y+35)))
        nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x-290,center_x-205), y_coords=(center_y-35,center_y+35)))
        central_nasal = number_filter(get_Object_num(circle_image, x_coords=(center_x-155,center_x-75), y_coords=(center_y-35,center_y+35)))
    table_image = image[3117:3516,1568:2973]
    if (eye=='OD'):
        volume = number_filter(get_Volume_Object(table_image, x_coords=(1040,1180), y_coords=(210,285)))
        avg_thickness = number_filter(get_Thickness_Object(table_image, x_coords=(1040,1180), y_coords=(305,390)))
    else:
        volume = number_filter(get_Volume_Object(table_image, x_coords=(1220,1390), y_coords=(200,295)))
        avg_thickness = number_filter(get_Thickness_Object(table_image, x_coords=(1220,1390), y_coords=(300,395)))
    report_row = {
    'Filename' : os.path.basename(image_path),
    'Name' : name,
    'Birthdate' : birthdate,
    'Gender' : gender,
    'Exam_Date' : exam_date,
    'Exam_Time' : exam_time,
    'Eye' : eye,
    'Signal_Strength' : signal_strength,
    'Superior' : superior,
    'Central_Superior' : central_superior,
    'Nasal' : nasal,
    'Central_Nasal' : central_nasal,
    'Inferior' : inferior,
    'Central_Inferior' : central_inferior,
    'Temporal' : temporal,
    'Central_Temporal' : central_temporal,
    'Central' : central,
    'Volume' : volume,
    'Avg_Thickness' : avg_thickness,
    'Fovea' : fovea
    }
    #print(report_row)
    #cv2.imshow('test', circle_image); cv2.waitKey(0)
    report_list.append(report_row)
    #if show_img:
    #    cv2.imshow(filename, image); cv2.waitKey()
    return report_list

def flag_numeric(val, pmean, thresh):
    try:
        val = float(val)
    except:
        color = '#FFCCCB'
        return 'background-color: {}'.format(color)
    if (math.isnan(val)):
        color = 'yellow'
    elif ((val > (pmean+thresh)) | (val < (pmean-thresh))):
        color = 'yellow'
    else:
        return False
    return 'background-color: {}'.format(color)

def flag_gender(val):
    if val.lower() not in {'male', 'female'}:
        color = '#FFCCCB'
    else:
        return False
    return 'background-color: {}'.format(color)
    
def flag_eye(val):
    if val.lower() not in {'od', 'os'}:
        color = '#FFCCCB'
    else:
        return False
    return 'background-color: {}'.format(color)

def flag_signal(val):
    try:
        val = int(val)
    except:
        color = '#FFCCCB'
        return 'background-color: {}'.format(color)
    if ((val > 10) | (val < 0)):
        color = '#FFCCCB'
    else:
        return False
    return 'background-color: {}'.format(color)

def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}-{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}-{}{}".format(filename, i, file_extension)
    return new_fname

@app.route("/", methods=["GET", "POST"])
def upload_image():
    return render_template("index.html")

@celery.task(bind=True)
def bigone(self):
    i=0
    k=0
    self.update_state(state='PROGRESS',meta={'current': i, "processed": k})
    INPUT_PATH= './Input/'
    #delete_files_in_directory(INPUT_PATH)
    for filename in sorted(os.listdir(INPUT_PATH)):
        if filename.endswith('.pdf') or filename.endswith('.png'):
            os.remove(os.path.join(INPUT_PATH, filename))

    objects = s3_client.list_objects(Bucket='secret')
    for obj in objects['Contents']:
        print(obj['Key'])
        if obj['Key'] == "secret":
            s3.delete_object(Bucket="secret", Key=obj['Key'])
        else:
            s3.download_file(Bucket="secret", Key=obj['Key'], Filename=os.path.join(INPUT_PATH,obj['Key']))
            s3.delete_object(Bucket="secret", Key=obj['Key'])
            k+=1
            self.update_state(state='PROGRESS',meta={'current': i, "processed": k}) 
    # if not os.path.exists(INPUT_FILES):
    #     os.makedirs(INPUT_FILES)
    # delete_files_in_directory(INPUT_FILES)
    for filename in sorted(os.listdir(INPUT_PATH)):
        print(filename)         
    INPUT_FILES = [os.path.join(INPUT_PATH, pdf_file) for pdf_file in os.listdir(INPUT_PATH)]
    print(INPUT_FILES)
    if os.path.join(INPUT_PATH, '.DS_Store') in INPUT_FILES: INPUT_FILES.remove(os.path.join(INPUT_PATH, '.DS_Store'))
    OUTPUT_PATH = './PNG/'
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    for filename in sorted(os.listdir(OUTPUT_PATH)):
        if filename.endswith('.png') or filename.endswith('.pdf'):
            os.remove(os.path.join(OUTPUT_PATH, filename))
    for filename in sorted(os.listdir(INPUT_PATH)):
        file_path = os.path.join(INPUT_PATH, filename)
        if file_path.endswith('.pdf'):
            images = convert_from_path(file_path, dpi=500)
            new_filename = os.path.join(OUTPUT_PATH, filename[:-4] + '.png')
            images[0].save(new_filename, 'PNG')
        elif file_path.endswith('.png'):
            images = Image.open(file_path)
            new_filename = os.path.join(OUTPUT_PATH, filename)
            images.save(new_filename, 'PNG')
    INPUT_PATH = './PNG/'
    INPUT_FILES = [os.path.join(INPUT_PATH, png_file) for png_file in os.listdir(INPUT_PATH)]
    if os.path.join(INPUT_PATH, '.DS_Store') in INPUT_FILES: INPUT_FILES.remove(os.path.join(INPUT_PATH, '.DS_Store'))
    OUTPUT_FILE = 'secret
    filenames = sorted(os.listdir(INPUT_PATH))
    print(filenames)
    patient_segmentations = []
    start_time = time.time()
    for filename in filenames:
        if filename.endswith('.png'):
            image_path = os.path.join(INPUT_PATH, filename)
            scan_type = get_ScanType(image_path)
            if scan_type == 'ODOS': # Binocular scan with both eyes included
                report = extract_Binocular_Report(image_path, 'OD', one_eye_recorded=False, show_img=False, verbose=False)
                patient_segmentations.extend(report)
                report = extract_Binocular_Report(image_path, 'OS', one_eye_recorded=False, show_img=False, verbose=False)
                patient_segmentations.extend(report)
            elif 'OD' in scan_type or 'OS' in scan_type: # Binocular with only one eye included
                report = extract_Binocular_Report(image_path, scan_type, one_eye_recorded=True, show_img=False, verbose=False)
                patient_segmentations.extend(report)
            else: # Monocular scan with one eye included
                mono_eye = get_monocular_eye(image_path)
                if mono_eye == 'None':
                    print('The following monocular file does not have data for either eye: ', filename)
                else:
                    report = extract_Monocular_Report(image_path, mono_eye, show_img=False, verbose=False)
                    patient_segmentations.extend(report)
            i+=1
            self.update_state(state='PROGRESS',meta={'current': i, "processed": k})


    dataframe = pd.DataFrame(patient_segmentations)
    print(dataframe)
    dataframe.to_excel('secret', index=False)
    INPUT_FILE = 'secret'
    OUTPUT_FILE = 'secret'
    df = pd.read_excel(INPUT_FILE, dtype='str')
    styler = df.copy().style
    styler.applymap(flag_gender, subset='Gender')
    styler.applymap(flag_eye, subset='Eye')
    styler.applymap(flag_signal, subset='Signal_Strength')
    numeric_params = ['Superior', 'Central_Superior', 'Nasal', 'Central_Nasal', 'Inferior',
    'Central_Inferior', 'Temporal', 'Central_Temporal', 'Central', 'Volume', 'Avg_Thickness']
    for param in numeric_params:
        df[param] = pd.to_numeric(df[param], errors='coerce')
        param_mean = df[param].mean()
        param_std = df[param].std()
        threshold = 3 * param_std
        styler.applymap(flag_numeric, pmean=param_mean, thresh=threshold, subset=param)
    styler.to_excel(OUTPUT_FILE, index=False)
    s3.upload_file(Bucket='secret', Filename = OUTPUT_FILE, Key = 'secret')
    
    return


@app.route("/results", methods=["GET", "POST"])
def results():
    if request.files:
        pdf = request.files["Upload"]
        if pdf:
            INPUT_PATH= './Input/'
            files = request.files.getlist("Upload")
            for file in files:
                s3.upload_fileobj(file,'secret', file.filename)
            task = bigone.apply_async()
            #celery.send_task('app.bigone')
    return redirect(url_for("download", task_id = task.id))

@app.route("/download/<task_id>", methods=["GET", "POST"])
def download(task_id):
    task = bigone.AsyncResult(task_id)
    i = 0
    status = 3
    k = 0
    if task.state == 'PENDING':
        # job did not start yet
        status = 0
    elif task.state != 'FAILURE':
        if 'SUCCESS' in task.status:
            status = 2

        else:
            i = task.info["current"]
            if i == 0:
                status = 4
                k = task.info["processed"]
    
            else:
                status = 1
        #print(task.status)
    else:
        # something went wrong in the background job
        status = 3


    return render_template("download.html", status = status, i = i, k = k)

@app.route("/return-file", methods=["GET", "POST"])
def return_file():
    s3.download_file(Bucket="secret", Key='secret', Filename='secret')
    s3.delete_object(Bucket="secret", Key='secret')
    return send_file("secret")
