## Brier Score Test Performed

Test performed on Logistic Regression classifier, and SVC with isotonic and sigmoid calibration.
This test conclude the use of Logistic Regression as classifier with hard cluster for KULeuven data.

1. Logistic
⋅⋅⋅* Brier: 0.378
⋅⋅* Precision: 0.513
⋅⋅* Recall: 0.347
⋅⋅* F1: 0.405

2. SVC:
⋅⋅* Brier: 0.435
⋅⋅* Precision: 0.515
⋅⋅* Recall: 0.343
⋅⋅* F1: 0.403

3. SVC + Isotonic:
⋅⋅* Brier: 0.399
⋅⋅* Precision: 0.515
⋅⋅* Recall: 0.342
⋅⋅* F1: 0.402

4. SVC + Sigmoid:
⋅⋅* Brier: 0.252
⋅⋅* Precision: 0.420
⋅⋅* Recall: 0.421
⋅⋅* F1: 0.378
		
![alt text](https://raw.githubusercontent.com/rxgranda/uncertaintyServerComponents/master/doc/calibration_test/Hard_classification/calibration_SVC_hard_clustering.png)
