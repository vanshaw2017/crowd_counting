```

```

the model I used in this implementation  was inspired by MCNN from CVPR2016 paper 
[Single Image CrowdCounting via Multi-Column Convolutional Neural Network](https://ieeexplore.ieee.org/document/7780439)

and I download my data from kaggle  

[crowd counting](https://www.kaggle.com/fmena14/crowd-counting)

At the top of the model in the original paper, 1x1 conv filter was used to  to learn the map from the raw pixels to the density maps map the conv feature block to density image. So you need to get you data prepared in the form of density maps. This conversion is too complicated when there are not many people in each frame.So after I extracted the multi-granularity features with multi-column convolution, I used regression to calculate the population density.