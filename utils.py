import glob
import cv2 # pip install cv2
from shutil import copyfile
import random
import numpy as np
import os
import pandas as pd
import apiai
import keras # pip install keras
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from gtts import gTTS
import playsound
import speech_recognition as sr #pip install SpeechRecognition

""" *****************   DEFINE CLASSIFIERS *****************  """
faceDet_haar = cv2.CascadeClassifier("libraries\\haarcascade_frontalface_default.xml")
faceDet_LBP = cv2.CascadeClassifier("libraries\\lbpcascade_frontalface.xml")

""" *****************   DEFINE FUNCTIONS   ***************** """

def file_count(path):
    """
    Summary: This function counts number of  files in folder
    @P1: path of the folder
    Returns: number of elements in the folder
    """
    return len([f for f in os.listdir(path)])

def delete_from_os(file):
    """
    Summary: This function removes files from OS
    @P1: file: path
    Returns: void method
    """
	# if no faces, delete image from OS
    try:
        os.remove(file)
    except OSError as e:
        print("Failed with:", e.strerror)
        #print(" System remove error at file: ", file)

def shuffle_files(path):
    """
    Summary: This function shuffels the files into 80% training, 20% prediction
    @P1: path
    Returns: training, prediction
    """
    files = glob.glob(path)
    #print("files -> ", files)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.20):] #get last 20% of file list

    return training, prediction

""" IDENTITIES & EMOTIONS """

def detect_face(img_path):
    """
    Summary: Detect face on file, if not detected, we delete imagen with 
    @P1: img_path: path
    Returns: Face of the image
    """
    #read image
    img = cv2.imread(img_path)

    try: 
        #convert the test image to gray scale as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        gray = cv2.imread(img_path, 0)

    
    #let's detect multiscale images(some images may be closer to camera than others)
    #result is a list of faces
    faces = faceDet_LBP.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    #if no faces are detected then return original img
    if (len(faces) == 0):
        delete_from_os(img_path)
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    x, y, w, h = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def prepare_emotion_dataset():
    """
    Summary: this method detects the face of an image, resizes it (based on the face) 
    and saves it in the folder: dataset/emotion
    Returns: void method
    """
    print("preparing dataset")
    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir("Images\\examples\\")

    # Get all files
    for dir_name in dirs:
        path = "Images\\examples\\" + dir_name + "\\*.jpg"
        files = glob.glob(path)
        print("Folder: ", dir_name)

        for f in files:
            # Detect faces
            face, rect = detect_face(f)

            try:
                print("escribo en dataset")
                #Resize face so all images have same size
                out = cv2.resize(face,(350, 350))
                # We detect the number of elements in the folder and we name the file with it
                # This is a more automated way in case we arent resizing all images at the same time
                filenumber = file_count("Images\\dataset\\%s\\" %(dir_name))
                #filenumber = filenumber+1
                cv2.imwrite("Images\\dataset\\%s\\%s.jpg" %(dir_name, filenumber), out) #Write image
            except:
                print("CV rewrite error")
                pass #If error, pass file

def prepare_sets_identity__emotions_train(data_folder_path, is_emotions, total_training):
    """
    Summary: the function reads all the data inside the path  and returns all faces found with 
    their corresponding label
    @P1: data_Folder_path: Folder where you have all the folders with training
    @P2: is_emotions: if true, there's no face detection
    @P3: Total training: we are training with all data, no predictions
    Returns: faces_training, labels_training, faces_prediction, labels_prediction
    """

    #get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)
    print(dirs)

    #list to hold all subject faces
    faces_training = []
    #list to hold labels for all subjects
    labels_training = []
    #list to hold all subject faces
    faces_prediction = []
    #list to hold labels for all subjects
    labels_prediction = []
    training = []
    prediction = []

    #let's go through each directory and read images within it
    for dir_name in dirs:
        print("Folder: ", dir_name)

        #build path of directory containing images for current subject subject
        #sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path  + dir_name + "\\*"
        #print(subject_dir_path, " o ", data_folder_path )
        #print("antes del shuffle")

        if total_training == False:
            training, prediction = shuffle_files(subject_dir_path)
        else: 
            training = glob.glob(subject_dir_path)

        label = int(dir_name)

        #Append data to training and prediction list, and generate labels 0-7
        for item in training:        

            if is_emotions:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                faces_training.append(gray) #append image array to training data list
                labels_training.append(label)
            else:
                face, rect = detect_face(item)
                if face is not None:
                    #add face to list of faces
                    faces_training.append(face)
                    #add label for this face
                    labels_training.append(label)

        for item in prediction: #repeat above process for prediction set

            if is_emotions:
                image = cv2.imread(item) #open image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
                faces_prediction.append(gray) #append image array to training data list
                labels_prediction.append(label)
            else:
                #detect face
                face, rect = detect_face(item)
                if face is not None:
                    #add face to list of faces
                    faces_prediction.append(face)
                    #add label for this face
                    labels_prediction.append(label)

    return faces_training, labels_training, faces_prediction, labels_prediction

def train_identities_emotions(label, recognizer, train, is_emotions, total_training):
    """
    Summary: this function trains identities or emotions
    @P1: Label [can be identity or emotion]
    @P2: recognizer, fish_face_recognizer or LBPRecognizer
    @P3: train: bool
    @P4: is_emotion: bool, used in prepare sets
    @P5: total_training, bool, used in prepare sets
    Returns: void method
    """ 
    path =""
    xml_path =""

    if label == "identities":
        print ("Identity recognizer: Training lbp face recognizer")
        path = "Images\\people_detection\\"
        xml_path = "libraries\\identities.xml"
    else:
        print ("Emotion recognizer: Training fish face recognizer")
        path = "Images\\dataset\\"
        xml_path = "libraries\\emotionRecognition.xml"

  
    if train:

        training_data, training_labels, prediction_data, prediction_labels = prepare_sets_identity__emotions_train(
                                                                                path, is_emotions, total_training)
        
        print ("size of training set is:", len(training_labels), "images")
        recognizer.train(training_data, np.asarray(training_labels))
        print ("predicting classification set")
        recognizer.save(xml_path)

        cnt = 0
        correct = 0
        incorrect = 0
        score = ""

        if len(prediction_data) > 0:
            for image in prediction_data:
                # Predict image
                pred, conf = recognizer.predict(image)
                if pred == prediction_labels[cnt]:
                    correct += 1
                    cnt += 1
                else:
                    incorrect += 1
                    cnt += 1
            score = "Score: " + str((100*correct)/(correct + incorrect))
            print(score)
        #else:
            #print("El score no está disponible para un entrenamiento de los datos al 100%")
    else:
        print ("Retrieving recognizer from: ", xml_path)
        recognizer.read(xml_path)
              
def predict_identity_emotion(option, recognizer, list_identities_emotions, face):
    """
    Summary: this function predicts identities or emotions. 
            If identities, we have to detect the face.
            If emotions, face is already given
    @P1: option: [identities or emotions]
    @P2: recognizer: [lbp_face_recognizer or fish_face_recognizer]
    @P3: list_identities_emotions [list of identities or emotions]
    @P4: Face: [None for identities, face for emotions]
    Returns: label, conf
    """ 
    print("Predicting ", option, "...")
    #predict the image using our face recognizer 

    face =cv2.resize(face, (350, 350)) 
    label, conf = recognizer.predict(face)
    #get name of respective label returned by face recognizer
    label_text = list_identities_emotions[label]
    conf = round(conf, 2)
    #print("Confidence for: " + label_text + ": " + str(conf) + "%")
    return label_text, conf


""" chat_bot """

def chat_bot(text):
    """
    Summary: this function returns a message from chatbot
    @P1: text: input text by user
    Returns: chatbot output
    """

    CLIENT_ACCESS_TOKEN = 'token'
    ai = apiai.ApiAI(CLIENT_ACCESS_TOKEN)
    request = ai.text_request()
    request.lang = 'es'
    #request.session_id = "c987a2be-7f31-4939-8aae-e31ca4d9dc52"
    request.query = text
    response = request.getresponse()
    aux = str(response.read())
    return str(aux[aux.find("speech"):].split("\\n")[0]).replace(",","")

def audio_to_text(audio_file_path):
    """ 
    Summary: This function transforms audio in text.
    @P1: audio_file_path: audio path
    Returns: text
    """
    # Initiate Speech Recognition 
    r = sr.Recognizer()
    # Load audio file
    audio_file = sr.AudioFile(audio_file_path)

    with audio_file as source:
        audio = r.record(source)
    user_input = r.recognize_wit(audio, "token")
    return user_input

def text_to_audio(chat_bot_input):
    """ 
    Summary: This function transforms text to audio
    @P1: chat_bot_input: 
    """
    # sound
    tts = gTTS(text=chat_bot_input, lang='es').save("Speech\\user_audio_output\\prueba.mp3")
    playsound.playsound("Speech\\user_audio_output\\prueba.mp3", True)
    delete_from_os("Speech\\user_audio_output\\prueba.mp3")
    

""" OBJECTS """

def check_cache_keras():
    """
    Summay: this function creates the cache dir & model dir if doesn't exist 
    Returns: void method
    """
    print("start check_cache_keras")
    cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    models_dir = os.path.join(cache_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    print("end check_cache_keras")

def get_df_objects(image_path, top, dangerous_objs, resnet):
    """
    Summary: this function returns a dataframe with all objects and if they are dangerous or not
    @P1: image_path
    @P2: top: number of recognized objects we want to show
    @P3: dangerous_obj: list of dangerous objects defined in main
    @P4: resnet
    """
    keras_img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    keras_img = keras.preprocessing.image.img_to_array(keras_img)
    x = keras.applications.resnet50.preprocess_input(np.expand_dims(keras_img.copy(), axis=0))
    preds = resnet.predict(x)
    decode_array = keras.applications.imagenet_utils.decode_predictions(preds, top=top)

    # dangerous NO = 0 | YES = 1
    objects_df = pd.DataFrame(columns=["object", "pred", "dangerous"])

    for i in range(top):
        obj = str(decode_array[0][i][1])

        objects_df.loc[i, "object"] = obj
        objects_df.loc[i, "pred"] = str(decode_array[0][i][2])

        if obj in dangerous_objs:
            objects_df.loc[i, "dangerous"] = True
        else:
            objects_df.loc[i, "dangerous"] = False       

    return objects_df

""" TRAINING ALL """

def training_all_data(fish_face_recognizer, lbp_face_recognizer, train_identities, train_emots, total_training):
    """ 
    Summary: this function trains all recognizers
    @P1: fish_face_recognizer
    @P2: lbp_face_recognizer
    @P3: train_identities: bool
    @P4: train_emots: bool
    @P5: total_training
    """

    """ TRAINING IDENTITIES  """
    print("************************* TRAINING IDENTITIES\n")

    # Train identities
    train_identities_emotions("identities", lbp_face_recognizer, train_identities, False, total_training )

    """" TRAINING EMOTIONS """
    print("\n************************ TRAINING EMOTIONS\n")
    # moves from examples to dataset
    #prepare_emotion_dataset()

    # Train emotions
    train_identities_emotions("emotions", fish_face_recognizer, train_emots, True, total_training)

    """ GET PRE-TRAINED OBJECTS """
    print("\n*********************** DOWNLOADING PRE-TRAINED OBJECTS\n")

    # checking the cache
    check_cache_keras()
    # Downloading data from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5
    resnet = ResNet50(weights='imagenet')
    print("resnet downloaded", "\n")

    return resnet
