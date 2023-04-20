This is my code for classification with MNIST data

+ dataset(mnist) is available in link: https://drive.google.com/drive/u/0/folders/113e-7OW1r2rCnDZx-ma23yR13K3spamT - Please download MNIST into project
+ file train_eval.py for training loop and evaluate accuracy
+ file predict.py for predicting your image 
+ folder 'utils' contain some neccessary function like loss function, optimizer, load_data function, class dataset, plot_dataset_function, open_function...

How to predict your dataset?
+ easily, first you can put your test_set in folder 'data'
+ second, please change the index in the line 20 (__getitem__ function)
+ third, you can change your model if you want

How to train your custom data?
+ first, you can put your dataset in folder 'data'
+ second, change path to dataset in line 9,10,23,24 folder dataset.py
+ may be you should modified the class Dataset

The accuracy: accuracy.png


HOW YOU ENJOY IT!
