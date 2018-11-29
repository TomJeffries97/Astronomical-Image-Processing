#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 11:18:37 2018

@author: Francisco Costa & Tom Jeffries

Code Version 1: 
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

hdulist = fits.open("A1_mosaic.fits")

image_data = hdulist[0].data
N = len(image_data)
M = len(image_data[0])      
mask_image = np.ones((N,M))

def fakeimage():
    fake_image = np.zeros((10,10))
    fake_image[3][6] = 69
    fake_image[3][3] = 58
    fake_image[6][3] = 21
    fake_image[7][4] = 34
    fake_image[7][5] = 13
    fake_image[6][6] = 27
    return(fake_image)
    


            
def Findpeaks(count,bins):
    adjustment = (bins[1]-bins[0])/2
    newbins=[]
    newcount = []
    for i in range(len(bins)-1):
        if count[i] > 0.001:
            newbins.append(bins[i]+adjustment)
            newcount.append(count[i])
    plt.figure(4)
    plt.clf
    plt.plot(newbins,newcount,'o', label = 'measured vales')
    plt.title('Background intensities')
    return(newcount,newbins)
        
        
def row_clean(k): #n is row, m is colom
    for i in range(150):
        for j in range(len(mask_image[0])):
            mask_image[k+i][j] = 0

def colom_clean(l):
    for i in range(len(mask_image)):
        for j in range(150):
            mask_image[i][j+l] = 0
            
def area_clean(xmin,xmax,ymin,ymax):
    for i in range(xmin,xmax):
        for j in range(ymin,ymax):
            mask_image[j][i] = 0

def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))





def clean():
    row_clean(0)
    row_clean(N-150)
    colom_clean(M-150)
    colom_clean(0)
    area_clean(1214,1661,2988,3436)
    area_clean(1427,1448,0,4611)
    area_clean(1103,1651,422,475)
    area_clean(1019,1702,313,374)
    area_clean(1390,1476,199,278)
    area_clean(1027,1044,424,451)
    area_clean(728,838,3204,3418)
    area_clean(865,950,2222,2357)
    area_clean(928,1028,2703,2834)
    area_clean(2102,2166,3707,3803)
    area_clean(2103,2159,2278,2336)
    area_clean(2064,2116,1400,1453)
    area_clean(535,688,4072,4123)
    area_clean(1440,1482,4010,4055)
    
    
plt.figure(1)
plt.imshow(image_data, cmap = 'prism')

clean()
fig = image_data*mask_image
plt.figure(3)
plt.imshow(fig, cmap = 'hot')

plt.figure(2)
NBINS = 1000
count, bins, ignored = plt.hist(fig.flatten(), NBINS, normed = True, range = [3335,3600]) 

max_values = []
for i in range(len(image_data)):
    value = max(image_data[i])
    for j in range(len(image_data[0])):
        if image_data[i][j] == value:
            max_values.append([value, i ,j])
            
            



y1, x1 = Findpeaks(count,bins)
params,extras = curve_fit(gaus,x1,y1, p0 = [0.2,3425,12])
print(params)
x = []
y = []
for i in np.linspace(3300,3600,10000):
    x.append(i)
    value = gaus(i,params[0],params[1],params[2])
    y.append(value)

plt.figure(4)
plt.xlabel('intensities')
plt.ylabel('counts')
plt.plot(x,y,label = 'gaussian fit')
plt.legend()    