# import the necessary packages
from imutils import paths
import face_recognition
from sklearn import svm
import argparse
import pickle
import cv2
import os
# import joblib
'''
This script expect images of one face only. It will convert all the face in the training folder to encoudings and save for later for face recognition.
When svm argument is true, the script will also train a svm model and save it for deployment.
'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--train_path", default="./train", help="path of images used for training")
ap.add_argument("-e", "--encodings",  default="encodings.pickle", help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection_method", default="HOG", help="select method for face detection, HOG (default) or cnn")
ap.add_argument("-c", "--svm",type=int, default=0, help="whether to create a svm classifier")


args = vars(ap.parse_args())

train_folder_path = args["train_path"]
train_images = list(paths.list_images(train_folder_path))

knownEncodings = []
knownNames = []

for (i, image) in enumerate(train_images):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,len(train_images)))
    name = image.split(os.path.sep)[-2] #get the name from the folder name
    # load the input image and convert it from BGR (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #reduce the size of the image for faster processing
    # scale_percent = 50 # percent of original size
    # width = int(image.shape[1] * scale_percent / 100)
    # height = int(image.shape[0] * scale_percent / 100)
    # dim = (width, height) 
    # image = cv2.resize(image,dim, interpolation = cv2.INTER_AREA)

    boxes = face_recognition.face_locations(image,model=args["detection_method"])
    # loop over the encodings
    if (len(boxes) == 1): # make sure there is only one face per images (This is reference encodings)
        # compute the facial embedding for the face
        encoding = face_recognition.face_encodings(image, boxes)[0]
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()

if (args["svm"]):
    print("[INFO] traning SVM model")
    clf = svm.SVC(gamma='scale')
    clf.fit(knownEncodings,knownNames)
    filename = 'svm_model.sav'
    pickle.dump(clf, open(filename, 'wb'))

