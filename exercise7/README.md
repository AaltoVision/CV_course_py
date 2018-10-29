## Exercise Round 7

For this exercise round, you should return the notebook ExerciseRound7.ipynb containing your source codes for Exercise 1 and your written answers to Exercises 2 and 3 (see part[1-3].ipynb). 

Note that for Exercises 2 and 3 you do not need to return any code, just provide written answers to the questions. <br>
First part of the tasks (Ex1-2) can be found in the ExerciseRound7.ipynb. For Ex3 (VGG practical) you need to look at part[1-3].ipynbs.




<b>For Ex3 (VGG practical) you need to install some packages and download some additional data:</b><br>

Additional libraries: <br>
Install the following two packages using your existing conda environment (like it was instructed <a href='https://github.com/AaltoVision/CV_course_py/blob/master/README.md'>here</a>)<br>

NOTE: Be sure to have your env activated before installing.<br>

```
conda install -c conda-forge pyflann # Approximate Nearest Neighbor matching
conda install line_profiler # Profile descriptor matching
```

Additional data: <br>
Download the following files into the folder './data_part3/' <br>

[imdb.npy](https://drive.google.com/open?id=1P8TjRIdwYtJHpm3l88v-8Gm1DbZSj-5P) <br>(checksum 1132d7d850fba611436eb74b43fd715b) <br> 
[sift_disc_vocab.npy](https://drive.google.com/open?id=1pMOcLj5AT4DiSzzUfejoTOZr5b1kp7aZ) <br> (checksum e6264c5b7c59d735ce92947add7cd636)<br>


Download the images used in the practical:
```
cd $exercise7_folder$ 
cd data
wget http://www.robots.ox.ac.uk/~vgg/share/practical-instance-recognition-2018a-data-only.tar.gz
tar -zxvf practical-instance-recognition-2018a-data-only.tar.gz
mv practical-instance-recognition-2018a/data/* .
```
