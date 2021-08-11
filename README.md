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
