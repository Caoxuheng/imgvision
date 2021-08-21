# Install
pip install imgvision
# Author
Xuheng Cao (caoxuhengcn@gmail.com)
# Example
## imgvision.spectra()
This Class is to create a standard CIE illuminance for the subsequent operation.  
>In[1]: &ensp;import numpy as np  
&emsp;&emsp;&emsp;import imgvision as iv  
&emsp;&emsp;&emsp;Illuminate_D65 = iv.spectra('d65')  
&emsp;&emsp;&emsp;Illuminate_A = iv.spectra('a')  
&emsp;&emsp;&emsp;Illuminate_D75 = iv.spectra('d75')  
&emsp;&emsp;&emsp;Illuminate_D50 = iv.spectra('d50')  

## imgvision.spectra.space()
The following example shows a mapping between a hyperspectral image with 31 bands from 400nm~700nm and the corresponding sRGB image.  
>In[2]:  &ensp;import matplotlib.pyplot as plt  
&emsp;&emsp;&emsp;sample_SI = np.random.rand(100,100,31)  
&emsp;&emsp;&emsp;sample_MI = Illuminate_D65.space(sample_SI,'srgb')  
&emsp;&emsp;&emsp;plt.imshow(sample_MI)  
&emsp;&emsp;&emsp;plt.show()  

## downsample 
The downsample function aims at downsampling with any scale factor for an image.  
>In[3]: &ensp;LR_sample_SI = iv.downsample(sample_SI,5)  
 &emsp; &emsp; &emsp;print(LR_sample_SI.shape)  
Out:    (20, 20, 31)  

## imgvision.color.rgb2hsv()
This function is to convert an RGB image (no matter how is it depth) to the HSV color space.  *hsv2rgb* excutes the contrary operation.  
>In[4]:&ensp; sample_HSV = iv.color.rgb2hsv(sample_MI)  
>In[5]: &ensp;sample_RGB =  iv.color.hsv2rgb(sample_HSV)  

## imgvision.distance.cosine()
Calculate the Cosine Similarity distance between two color images and return a 1-D vector, the vector include the Cosine Similarity distance of each element. This function supports the shape of input as a 2-D matrix (an RGB vector) and a 3-D tensor (a color image).

>In[6]:&ensp;a= np.random.rand(128,128,3)  
 &emsp;&emsp;&ensp;b= np.random.rand(128,128,3)  
 &emsp;&emsp;&ensp;dist = iv.distance.cosine(a,b)  
 &emsp;&emsp;&ensp;print(f'Shape: {dist.shape}\nMean distance: {dist.mean()}')  
Out:  Shape: (16384,)  
 &emsp; &emsp;Mean distance: 0.2055142334948004
 
## imgvision.cluster.cosine_predict(img,centre)
Predict clusters each pixel of a given image belongs to according to Cosine Similarity distance and given cluster center.
>In[7]:&ensp; img = np.random.rand(128,128,3)  
 &emsp; &emsp;&ensp;centre = np.array([ [0.3,0.5,0.1],  
  &emsp; &emsp; &emsp;&emsp; &emsp;&ensp;&emsp; &emsp; &emsp; &emsp;[0.8,0.1,0.3] ])  
 &emsp; &emsp;&ensp;cluster_id = cluster.cosine_predict(img,centre)  
 &emsp; &emsp;&ensp;print(cluster_id)  
Out: [1 1 1 ... 1 0 1]

