Read Me:

1) Train.IPYNB 
The jupyter notebook where training takes place. You can alter the hyper parametrs to look for the best performing model and save the corresponding weights into weights file.
This notebook requires the CLC_train.csv, CLC_test.csv to be present in the same folder.

2) Test.py
Script which loads the model (from the weights file) and evaluates on CLC_test.csv (required to be present in the same folder) and stores the predictions in submission_trials.csv under the column "Our prediction".

3) weights
The file which stores the state_dict of our best performing model.

4)submission_csv.csv
The csv files which stores the predictions of our best performing model(from the weights file)

5)Report.pdf
Comprehensive description of our experiments and our findings along with the results.