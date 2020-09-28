使用说明：

1. 需要在dataset文件夹里存放NUS-WIDE、VOC2012、COCO的原始图片数据，如果使用本地文件夹，需对应修改unseen_img_dir，seen_img_dir的参数。

2. 根目录下的models文件夹用于存放模型；如果想通过我们训练好的模型生成hash编码，则需要将我们的模型下载到model夹。

3. word2vec模型使用的是glove模型，更换标签路径可自行提取word向量。(相关代码在Word2vec文件夹中)


---
How to use the codes?

1  download the three public datasets and put them in the NUS-WIDE/VOC2012/COCO sub-folders. When you use some other datsets, you have to modify the path for the parameters unseen_img_dir and seen_img_dir.

2 if you want to produce the hash codes using our pre-trained model, you need to download our models and put them in the folder `models'.

3 for word2vec, the glove model is used. You can generate new word vectors by changing the path of the label file. ( codes for word2vec can be found in the `word2vec' folder)


