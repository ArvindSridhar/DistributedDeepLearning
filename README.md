# Distributed Deep Learning
Training a distributed neural network model
\nIdea: want to parallelize neural network computations
Take your dataset (training set), distribute it equally across 10 segments
Train the same neural network model separately on each segment
After 1 epoch (go through the whole data batch on that segment), merge the networks by averaging the weights
Possibly, explore weighted averages based on the loss of each model on its training batch
Basically, a way of "minibatching"
