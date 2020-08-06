# import the necessary packages
from imutils.video import VideoStream
import face_recognition
from sklearn import svm
import argparse
import imutils
import pickle
import time
import cv2
import numpy as np
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", default="encodings.pickle",
	help="path to serialized db of facial encodings")
ap.add_argument("-o", "--output", type=str, 
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
ap.add_argument("-d", "--detection_method", type=str, default="HOG",
	help="face detection model to use: either `hog` or `cnn`")
ap.add_argument("-rm", "--recognition_method", type=str, default="vote",
	help="face recognition method to use: either `min` or `vote`")
ap.add_argument("-svm", "--svm_path", default="svm_model.sav",type=str, help="path for the classfier")
ap.add_argument("-t", "--tolerance",type=float, default=0.40,help="tolerance for compare encoding distance, the less the more similar the faces need to be")
args = vars(ap.parse_args())

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open(args["encodings"], "rb").read())
if (args["recognition_method"]=="svm"):
    clf = pickle.loads(open(args["svm_path"],'rb').read())
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

boxes = []
encodings = []
process_this_frame = True

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    resize_ratio=2
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    small_frame = cv2.resize(frame, (0, 0), fx=1/resize_ratio, fy=1/resize_ratio)
   
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    boxes = face_recognition.face_locations(rgb_small_frame,
        model=args["detection_method"])
    encodings = face_recognition.face_encodings(rgb_small_frame, boxes)

    # print(len(encodings))
    
    
    names = []
    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        
        if (args["recognition_method"]=="vote"):
            matches = face_recognition.compare_faces(data["encodings"],
                encoding,tolerance=args["tolerance"])
            name = "Unknown"
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
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
                # update the list of names
                # print(counts)
            names.append(name)
        if (args["recognition_method"]=="min"):
            name = "Unknown"
            face_distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match_index = np.argmin(face_distances)
            print(min(face_distances))
            if min(face_distances) < args["tolerance"]:
                name=data["names"][best_match_index]
            names.append(name)
        if (args["recognition_method"]=="svm"):
            name = clf.predict([encoding])
            print(clf.decision_function([encoding]))
            names.append(*name)

        
        

    #process_this_frame = not process_this_frame

    for ((top, right, bottom, left), name) in zip(boxes, names):
        # rescale the face coordinates
        top = int(top * resize_ratio)
        right = int(right * resize_ratio)
        bottom = int(bottom * resize_ratio)
        left = int(left * resize_ratio)
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255,0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom + 35), (right, bottom), (0, 255,0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 30), font, 0.75, (255, 255, 255), 1)
    
    # if the video writer is None *AND* we are supposed to write
    #the output video to disk initialize the writer
    if writer is None and args["output"] is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 20,(frame.shape[1], frame.shape[0]), True)
    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame)

    # check to see if we are supposed to display the output frame to
    # the screen
    if args["display"] > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

#  # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
# # check to see if the video writer point needs to be released
if writer is not None:
    writer.release()