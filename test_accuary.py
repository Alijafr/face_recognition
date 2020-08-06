# import the necessary packages
from imutils import paths
import face_recognition
from sklearn import svm
import argparse
import pickle
import cv2
import os

'''
This script is used to test the correct prediction of face recongnition model.
Currently in there are two methods for testing:
1-svm --> (require the path of svm model)
2- highest vote (default)--> (require the path of the encodings) 
'''

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--test_path", default="./test", help="path of images used for training")
ap.add_argument("-e", "--encodings",  default="encodings.pickle", help="path to serialized db of facial encodings")
ap.add_argument("-m", "--detection_method", default="HOG",help="select method for face detection, HOG (default) or cnn")
ap.add_argument("-t", "--tolerance", default=0.4,help="tolerance for compare encoding distance, the less the more similar the faces need to be")
ap.add_argument("-c", "--svm_path", type=str, help="path for the classfier")
args = vars(ap.parse_args())

if args["svm_path"] is  None:
    data = pickle.loads(open(args["encodings"],"rb").read())
else:
    clf = pickle.loads(open(args["svm_path"],'rb').read())


test_folder_path = args["test_path"]
test_images = list(paths.list_images(test_folder_path))
print("[INFO] start evalution")
correct = 0
for image in test_images:
    true_name = image.split(os.path.sep)[-2]
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height) 
    image = cv2.resize(image,dim)
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(image,model="HOG")
    encoding = face_recognition.face_encodings(image, boxes)
    # initialize the list of names for each face detected
    predicted_names = []
    # loop over the facial embeddings
    if args["svm_path"] == None:

        matches = face_recognition.compare_faces(data["encodings"],encoding[0],tolerance=args["tolerance"])
        predicted_name = "Unknown"
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                predicted_name = data["names"][i]
                counts[predicted_name] = counts.get(predicted_name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
            predicted_name = max(counts, key=counts.get)
    else:
        predicted_name = clf.predict(encoding)
    
    if (predicted_name==true_name):
        correct +=1

print("model got {}/{} correct ".format(correct,len(test_images)))
       
    
  