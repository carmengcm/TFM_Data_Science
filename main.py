import glob
import cv2 # pip install cv2
import utils as utl
import os


""" *****************   DEFINE EMOTIONS    ***************** """
emotions = ["neutral", "positive", "negative"]

""" *****************   DEFINE IDENTITIES    *****************  """
identities = ["Harrison_Ford", "Rajoy", "scott_wilson", "denzel_washington", "Shakira"]
user = "scott_wilson"

""" DEFINE DANGEROUS OBJECTS """
dangerous_objs = ["cleaver", "hatchet", "gun", "shotgun", "revolver", "pistol", "rifle",  
                    "assault_rifle", "knife", "blade", "scabbard", "power_drill", 
                    "bow", "shovel", "chain_saw", "pocket_knife" ]


# We create the fish_face_recognizer
fish_face_recognizer = cv2.face.FisherFaceRecognizer_create()
# create our LBPH face identiti recognizer 
lbp_face_recognizer = cv2.face.LBPHFaceRecognizer_create()


print("\nWelcome to the main app! \n")

""" TRAIN ALL DATA """
# We are going to define if we want to use an existing training for all the modules

print("Identities trained for this test --> ", identities, "\n")
print("Emotions trained for this test --> ", emotions, "\n")
print("Dangerous objects detected on this test --> ", dangerous_objs, "\n")


# If False, we retrieve model (previously saved)
train_identities = False
train_emotions = False
# If true, we train all the files, if False, we train 80% and predict 20% to get score
total_training = False


# Call training method
resnet = utl.training_all_data(fish_face_recognizer, lbp_face_recognizer, train_identities, train_emotions, 
                        total_training)
# Getting all testing files
image_paths = glob.glob("Images\\people_detection_test\\*")

# Loop all files
for img_path in image_paths:
    # Detect face for each file
    face, rect = utl.detect_face(img_path)
    # We are only making predictions if we recognize faces
    if face is not None:
        print("\n# 1. Detected face in: ", img_path)
        # Identiy person. We are only going to proceed if the person in the image is our User.
        identity, conf = utl.predict_identity_emotion("identity", lbp_face_recognizer, identities, face)
        print("# 2. Detected identity: ", identity, ", confidence: ", conf)

        # If confidence for the identity is > 60 and iddentity is user 60) & (identity == user) 
        if (conf >= 50) & (identity == user):
            #print(conf, "% ", "for ", identity, ". We found our user!")
            print(conf, "% ", "for ", identity, ". We found our user!")
            emotion, conf = utl.predict_identity_emotion("Emotions", fish_face_recognizer, emotions, face)
            
            print("\n# Detected emotion: ", emotion, ", confidence: ", conf)
            # If emotion is negative, we detect objects
            if emotion == "negative":
                # Prediction of the objects in the image
                # Dataframe with all the objects - column saying if dangerous or not
                print("\nDetecting dangerous objects...")
                objects_df = utl.get_df_objects(img_path, 10, dangerous_objs, resnet)

                print("\nWe found the following objects:\n", objects_df.object)

                # Detect if any objetct on our dangerous object list
                print("\nDangerous objects: \n", objects_df[objects_df["dangerous"] == True])

                """ BOT """
                user_input = utl.audio_to_text("Speech\\user_audio_input\\user_input.wav")
                # user_input = "Me encuentro mal"
                response = utl.chat_bot(user_input)
                print("\nUser input: ", user_input, ". \n Response: ", response )
                # Save and play audio
                utl.text_to_audio(response)
        else:
            print("Confidence not enough!")

                        
    else:
        print("Face not detected for ", img_path)

