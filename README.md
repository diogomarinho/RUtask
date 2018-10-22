# RUtask
IF you want to recreate the model just run


python3 python/data_engineer.py
The models will be saved at ./models/


If you want estimate if some datapoints are outliers
run:
python3 python/fd_predict.py -r example.txt (where example is file with IDS one per line)


some notes about library versions I'm using

  scikit-learn             0.20.0 
  pandas                   0.23.4
  gcsfs                    0.1.2
  numpy                    1.12.0
  seaborn                  0.9.0
