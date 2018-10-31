## Exercise Round 8

For this exercise round, you should return a report with written answers to the questions below. 

### Exercise 1. Face tracking example using KLT tracker
This exercise is based on the python face tracking demo ```faceTrackingDemo.py```. 

Run the example as instructed below and answer the questions.

a) Run the python example script with the command ```python faceTrackingDemo.py```<br>
b) Run the python example script with different input  ```python faceTrackingDemo.py --input=./obama.avi```<br>
c) What could be the main reasons why most of the features are not tracked very long in case b) above?<br>
d) How one could try to avoid the problem that the features are gradually lost? Suggest a one or more improvements.<br>
e) Voluntary task: Capture a video of your own face or of a picture of a face, and check that whether the tracking works for you. That is, replace ```obama.avi``` with your own video. 

### Exercise  2. Kanade-Lucas-Tomasi  (KLT)  feature  tracking  (Pen  &  paper  problem)
Read Sections 2.1 and 2.2 from the paper by Baker and Matthews (https://www.ri.cmu.edu/pub_files/pub3/baker_simon_2002_3/baker_simon_2002_3.pdf). Show that the Equation (10) in the paper gives the same solution as the equations on slide 25 of Lecture 7, when the geometric warping W 
(between the current frame and the template window in the previous frame) is a translation.
