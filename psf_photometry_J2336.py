#!/usr/bin/env python
# coding: utf-8

# # PSF Photometry on J2336

# In[1]:


#loading the image of J2336

import astropy
from astropy.io import fits
from astropy.io.fits import HDUList

import pathlib
import path

fits_image_J2336 = astropy.io.fits.open(pathlib.Path('Documents', 'research','unpacked_hst_data','J2336','ie6y03010_drz.fits'))

hdul = fits_image_J2336

image_data = fits.getdata('Documents/research/unpacked_hst_data/J2336/ie6y03010_drz.fits', ext=0)


# In[2]:


#visualize the image

import matplotlib
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

norm = simple_norm(image_data, 'asinh', percent=99.)
plt.imshow(image_data, norm=norm, origin='lower', cmap='viridis')
plt.colorbar()


# In[3]:


#find stars

from photutils.detection import find_peaks

peaks_tbl = find_peaks(image_data, threshold=0.01, box_size=(1000,1000), border_width=490)  
peaks_tbl['peak_value'].info.format = '%.8g'  # for consistent table output  
print(peaks_tbl)

#(x_peak, y_peak, peak_value)=(518, 553, 2.0802698) is the one I identified as being J23 on ds9


# In[4]:


#choosing stars

size = 25
hsize = (size - 1) / 2
x = peaks_tbl['x_peak']  
y = peaks_tbl['y_peak']  
mask = ((x > hsize) & (x < (image_data.shape[1] -1 - hsize)) &
        (y > hsize) & (y < (image_data.shape[0] -1 - hsize)))


#table of good star positions

from astropy.table import Table

stars_tbl = Table()
stars_tbl['x'] = x[mask]  
stars_tbl['y'] = y[mask]
print(stars_tbl[0])


# In[5]:


#extract stars

from astropy.nddata import NDData
nd_image_data = NDData(data=image_data) 

from photutils.psf import extract_stars
stars = extract_stars(nd_image_data, stars_tbl, size=25) 
print(stars[0].data)


# In[6]:


#plot the extracted stars

import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

nrows = 3
ncols = 3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20),
                        squeeze=True)
#cmaps = ['viridis', 'viridis', 'viridis']
ax = ax.ravel()
for i in range(nrows*ncols):
     norm = simple_norm(stars[i], 'log', percent=99.)
     ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis')
#     fig.colorbar(ax[i].imshow(stars[i], norm=norm, origin='lower', cmap='viridis'), ax=ax[i])


# In[7]:


#plotting J2336 alone
import numpy as np

norm = simple_norm(stars[0], 'log', 99.)
plt.imshow(stars[0], norm=norm, origin='lower', cmap='viridis')
plt.title('J2336 Original Image')
plt.colorbar()

print(np.max(stars[0])) #should be the same as peak_value for (x_peak, y_peak) = (518, 553)


# In[8]:


#finding image centroid
from photutils.centroids import centroid_com

com_centroid = centroid_com(stars[0].data, mask=None, oversampling=1)
print('From centroid_com:', com_centroid)


# In[9]:


get_ipython().run_line_magic('store', '-r epsf')

epsf.shape

#preparing the ePSF model
#from photutils.psf import prepare_psf_model
#prepared_epsf_model = prepare_psf_model(new_reshaped_scaled_epsf, xname=None, yname=None, fluxname=None)


# In[10]:


from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm

#setting parameters
sigma_psf=2.0
bkgrms = MADStdBackgroundRMS()
std = bkgrms(stars[0])
iraffind = IRAFStarFinder(threshold=3.5*std,
                          fwhm=sigma_psf*gaussian_sigma_to_fwhm,
                          minsep_fwhm=0.01, roundhi=5.0, roundlo=-5.0,
                          sharplo=0.0, sharphi=2.0, brightest=1)
daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
mmm_bkg = MMMBackground()
fitter = LevMarLSQFitter()
psf_model = epsf #insert my ePSF model

#setting quasar location
#psf_model.x_0.fixed = True
#psf_model.y_0.fixed = True
#pos = Table(names=['x_0', 'y_0'], data=com_centroid)

#performing photometry
from photutils.psf import BasicPSFPhotometry
photometry = BasicPSFPhotometry(group_maker=daogroup, bkg_estimator=mmm_bkg, psf_model=psf_model, 
                                fitter=LevMarLSQFitter(), fitshape=(11,11), finder=iraffind, aperture_radius = 5)
result_tab = photometry(image=stars[0]) #init_guesses=pos)
residual_image = photometry.get_residual_image()


# In[11]:


#imaging results
plt.subplot(1, 2, 1)
plt.imshow(stars[0], cmap='viridis', aspect=1,
           interpolation='nearest', origin='lower')
plt.title('Original Image J2336')
plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)

plt.subplot(1 ,2, 2)
plt.imshow(residual_image, cmap='viridis', aspect=1,
           interpolation='nearest', origin='lower')
plt.title('Residual Image J2336')
plt.colorbar(orientation='vertical', fraction=0.046, pad=0.04)


# In[12]:


#checking values
import numpy as np
print('The maximum of the original image:', np.max(stars[0]))
print('The maximum of the residual image:', np.max(residual_image))


# In[13]:


#important values
print(result_tab['x_0', 'y_0', 'flux_0'])
print(result_tab['x_0_unc', 'y_0_unc'])
print(result_tab['x_fit', 'y_fit', 'flux_fit', 'flux_unc'])

