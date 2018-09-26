# Distributed Deep Learning
**Overview**  
This is a simulation of training a distributed deep learning model over different segments of data in parallel, exploring techniques to get distributed training to achieve maximal accuracy while remaining efficient and lean.  
  
**Instructions for running**  
To simulate training simple deep NN models on each segment, run `python3 distributed_nn_training.py`. To simulate training CNN models on each segment, run `python3 distributed_cnn_training.py`  
Within each file, there are hyperparameters that you can tune:  
-The number of output classes (default 10 for MNIST)  
-The number of so-called "grand epochs" (default 1): this is the number of times we train each segment model fully on its portion of the dataset, and subsequently average the weights of the segment models to get the "merged" model. **Note: if you are interested in the ensemble techniques only and don't want to test the weight-average model technique, keep this as 1!** If 1, the segment models will be trained normally but not overwritten with averaged weights after training.  
-The batch size (default 100): how many examples are used to train at each iteration of gradient descent for each of the segment models as well as the ensemble models  
-The number of segments (default 10): program will divide up the 60,000 MNIST training examples equally among the n segments  
-The number of iterations on each segment (default 3): controls how many epochs we train each individual segment model for  
