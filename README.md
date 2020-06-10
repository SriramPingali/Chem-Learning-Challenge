# Chem-Learning-Challenge
Air pollution is a persistent problem of the world, which has been around for a while and will continue to exist in the future. It is a cause of a lot of problems and has extremely adverse effects. For curbing air pollution, its analysis is first necessary. For making predictions about the air quality, domain knowledge is essential, as are data analysis skills. This problem statement tests them all.

## Problem Statement
Participating teams are invited to propose and implement a multiclass classification technique for a given data set of air quality measurements. We expect a well-structured report detailing the approach used for classification and its implementation. Teams would be provided with the training data set to be used. Teams are expected to brainstorm, ideate, experiment, and code classification techniques to get the best results. The goal of this challenge is to create awareness about the applications of data analysis and machine learning in the chemical engineering domain, especially air quality analysis for air pollution control applications among the student community and provide them a platform to showcase their ideas and innovations.

## Files Description
1) Train.IPYNB 
The jupyter notebook where training takes place. You can alter the hyper parametrs to look for the best performing model and save the corresponding weights into weights file.
This notebook requires the CLC_train.csv, CLC_test.csv to be present in the same folder.

2) Test.py
Script which loads the model (from the weights file) and evaluates on CLC_test.csv (required to be present in the same folder) and stores the predictions in submission_trials.csv under the column "Our prediction".

3) weights
The file which stores the state_dict of our best performing model.

4) submission_csv.csv
The csv files which stores the predictions of our best performing model(from the weights file)

5) CLC_train.csv, CLC_test.csv are train and test data respectively.

6) Data_analysis.ipynb is the priminary data exploration and finding out relevant features and attributes.

7) Report.pdf Comprehensive description of our experiments and our findings along with the results.
