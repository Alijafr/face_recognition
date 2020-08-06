# face_recognition
This repo provide an fast face recognition system that can work with cpu (no need for gpu). The implementation is based on the well-known library *dlib* and *face_recognition*.
## Method
The face recognition step works as follow:
* 1-Find face in the image
* 2-Align face (not currently implemented)
* 3-Get the encoding of the face (a unique 128 vecotor representaion of face). In the repo, Openface is used.
* 4-Use a classification method to identifiy the face in the image. The available method are: min distance, highest vote,and svm model. 
## Scripts
There are 3 main scripts:
* train.py : expect a path of the *train* folder (The default is ./train). Inside this *train* folder, it should be folders of the people to be recognized.
For example /train/Ali etc. Furthermore, inside each person's folder, it should contain images of only this person so that model does not get confused. 
Lastly, there is an option to create a svm model, otherwise the encodings will be used for prediction.

*test_accuary.py: used to test the accuary of the model. It expect a path for *test* folder that has the same structure as the *train* folder. The default evalution is using the produced encodings and highest vote model. In case, you chose the svm model, you will need to provide the path of the file.
After runing the train.py, it can be found as *svm_model.sav*.

* web_cam.py: This is the main script. It will either use the encodings or the svm model generated from the ***train.py*** to recognize people in the image. There are 3 methods that implemented here: min distance, highest vote, and svm mode.
You can choose the method using *-rm* argument. 
