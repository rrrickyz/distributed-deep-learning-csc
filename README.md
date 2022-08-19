# distributed-deep-learning-csc

## Some problems that I need help with:

### 1. I cannot run command Horovodrun.

https://github.com/rrrickyz/distributed-deep-learning-csc/blob/main/building-error.md

### 2. I am using imagenet for training deep-learning models. For ResNet50, I noticed that the training accuracy is very low (9.3900e-04), and it changes
only a little, and the validation accuracy strangely stays the same (0.0010), no matter I use the pretrained weights 'imagenet' or I train the weight from scratch. 
At first, I thought the optimizer was not suitable, so I changed SGD to Adam, the training loss improved from **loss: nan** to **loss: 6.9080**, but it
only changed to **loss: 6.9079** in the following epoch, and the validation accuracy still stays the same. Learning rate does not matter either. 

The python script of the model: 

(I still have others unpushed to the github because of the conflicting repositories. Please wait until later!)

