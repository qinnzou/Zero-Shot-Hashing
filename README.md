# Zero-Shot-Hashing
Transductive Zero-Shot Hashing For Multi-Label Image Retrieval


This is the source code of Transductive Zero-Shot Hashing For Multi-Label Image Retrieval. We provide the codes, the partitioning of the datasets and the pretrained model.

Zou Q, Cao L, Zhang Z, Chen L, and Wang S, Transductive Zero-Shot Hashing For Multi-Label Image Retrieval, Under review, 2020.


## Download:

You can also download the **pretrained model** from the following link,   
https://pan.baidu.com/s/1y0A0rHaJcrbaPQzkfT_xNQ   
passcodes：abcd 

# Set up
## Requirements
PyTorch 1.0.0  
Python 3.6  
CUDA 10.0  
We run on the Intel Core Xeon E5-2630@2.3GHz, 64GB RAM and two GeForce GTX 1080Ti GPUs.

## Preparation
### Data Preparation (training/test image indice)
Download the three public datasets and put them in the dataset/NUS-WIDE, VOC2012, COCO sub-folders, respectively. When you use some other datasets, you have to modify the values of the parameters unseen_img_dir and seen_img_dir.  

In the folder `./data/', we have given the lists for partitioning the datasets for NUS-WIDE, VOC2012 and COCO. The indice of the training/test images can be found in the three subfolders.

For word2vec, the glove model is used. You can generate new word vectors by changing the path of the label file. ( codes for word2vec can be found in the `word2vec' folder)

### Pretrained Models
Pretrained models using PyTorch are available using the link below, including the eight models, which take the ResNet50 as the backbone network, and are trained on between the Nuswide and Voc datasets, with 12-, 24-, 36-, and 48-bits hash codes.   
You can download them and put them into "./models/".    
https://pan.baidu.com/s/1y0A0rHaJcrbaPQzkfT_xNQ   
passcodes：abcd 

## Training
Before training, you should set the right path to the datsaets you are working on, and then run:  
```
python train_alexnet.py
```
or  
```
python train_vgg.py
```
or  
```
python train_resnet.py
```

## Test
To evlauate the performance of a pre-trained model, please put the pretrained model listed above or your own models into "./models/" and change "pretrained_path" in config.py at first, then Choose the right model that would be evlauated, and then simply run:  
```
python eval_alexnet.py
```
or  
```
python eval_vgg.py
```
or  
```
python eval_resnet.py
```


