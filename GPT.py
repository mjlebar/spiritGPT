import torch 
import torch.nn as nn
from torch.nn import functional as F


#set hyperparameters

#### Process text

#read in the file

#make the vocabulary

#make the encoding and decoding (text to number, number to text)

#make up the blocks (including padding + start & end tokens)

#split into train, validation, test sets


###set up NN architecture

#embedding class - pytorch

#position embedding class

#head class

#multi-head layer class

#layernorm class - pytorch

#block class

#model class:
#def
#forward


###train

#initialize loss function and optimizer

#loop through batches input and targets from train:
#get predictions from model
#get loss
#zero the gradient
#backpropagate
#take a step
#every 500 steps or the second to last step: output training loss vs validation loss


###output:
#use the model's predictions to output text