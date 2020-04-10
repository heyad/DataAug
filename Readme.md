### Sing Language Recognition 

This **[Notebook](Multi-Class-Sing.ipynb)** aims at introducing ImageDataGenerator, and show how it can be used to train a model to recongise sign language in a set of images. You will need to download the dataset from Kaggle https://www.kaggle.com/datamunge/sign-language-mnist and save it in your local drive. The use of `ImageDataGenerator` enable us to do data augmentation which is an essential task in many machine learnign algorithms. For more details see keras docs https://keras.io/preprocessing/image/. 

You can download the [Notebook](Multi-Class-Sing.ipynb) eithe by clicking Clone and Download button (top right) link or using command line: 
```
$ git clone https://github.com/heyad/DataAug.git
```

#### Improve Results 


* Add more conv/pooling layers 
* Change the 'ImageDataGenerator' and see the impact on the performance of the model

```python

train_datagen = ImageDataGenerator(
      rescale = 1./255.,
      rotation_range=5,
      width_shift_range=0.13,
      height_shift_range=0.3,
      fill_mode='nearest')
    
```



![png](figures/output_5_0.png)



