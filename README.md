# face_recognition
This repo provides a fast face recognition system that can work with cpu (no need for gpu). The implementation is based on the well-known libraries *dlib* and *face_recognition*.
## Method
The face recognition step works as follow:
* 1-Find face in the image
* 2-Align face (not currently implemented)
* 3-Get the encoding of the face (a unique 128 vecotor representaion of face). In this repo, Openface is used.
* 4-Use a classification method to recognize the face in the image. The available methods are: min distance, highest vote,and svm model. 
## Scripts
There are 3 main scripts:
* train.py : expect a path of the *train* folder (The default is ./train). Inside this *train* folder, it should contain folders of the people to be recognized.
For example /train/Ali. Furthermore, inside each person's folder, it should contain images of only this person so that model does not get confused. 
Lastly, there is an option to create a svm model, otherwise the encodings will be used for prediction.

*test_accuary.py: used to test the accuary of the model. It expects a path for *test* folder that has the same structure as the *train* folder. The default evalution is using the produced encodings and highest vote model. In case, you chose the svm model, you will need to provide the path of the file.
After runing the train.py, it can be found as *svm_model.sav*.

* web_cam.py: This is the main script. It will either use the encodings or the svm model generated from the ***train.py*** to recognize people in the image. There are 3 methods that implemented here: min distance, highest vote, and svm mode.
You can choose the method using *-rm* argument. 

## Installation
* 1-create a new environment 
```
conda create -n face_recognition python=3.6
conda activate face_recognition
```
* 2- Install dependencies
```

```
