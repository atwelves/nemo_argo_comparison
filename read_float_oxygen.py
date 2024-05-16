# script to read float oxygen without argopy

import xarray as xr
import numpy as np
import pandas as pd
import netCDF4 as nc
from netCDF4 import Dataset
import gsw
import cmocean

ds = xr.open_dataset('GL_PR_PF_3902106.nc')
print(ds)
oxy = np.transpose(ds.DOX2_ADJUSTED.values)
print(np.shape(oxy))
time = ds.TIME.values
long = ds.LONGITUDE.values
lat  = ds.LATITUDE.values
lat = np.tile(lat,(162,1))
long = np.tile(long,(162,1))
pres = np.transpose(ds.PRES.values)
depth = -gsw.conversions.z_from_p(pres,np.mean(lat))
print(np.shape(lat))
print(np.shape(long))
dates = pd.DatetimeIndex(time)
yrs = np.tile(dates.year,(162,1))
print(np.argmin(yrs[yrs==2020]))
mts = np.tile(dates.month,(162,1))
print(np.shape(mts))
print(mts)
print(dates.day)
oxy[np.isnan(oxy)] = -999
depth[np.isnan(depth)] = -999
oxy[yrs!=2020]=np.nan
oxy[mts>4]=np.nan
depth[yrs!=2020]=np.nan
depth[mts>4]=np.nan
print(np.shape(depth))
dates = np.tile(dates,(162,1))
dates = dates[np.isfinite(oxy)]
lat = lat[np.isfinite(oxy)]
long = long[np.isfinite(oxy)]
oxy = oxy[np.isfinite(oxy)]
depth = depth[np.isfinite(depth)]
oxy = np.reshape(oxy,(162,-1))
oxy = oxy/1000
depth = np.reshape(depth,(162,-1))
print(np.shape(oxy))
dates = np.reshape(dates,(162,-1))
lat   = np.reshape(lat,(162,-1))
long   = np.reshape(long,(162,-1))
oxy[oxy==-0.999]=np.nan
depth[depth==-999] = np.nan
# write to file
ncfile = Dataset('float_oxygen.nc', mode='w')
print(ncfile)
print('depth')
print(np.shape(depth))

y = ncfile.createDimension('y',np.size(oxy,0))
x = ncfile.createDimension('x',np.size(oxy,1))

#dpt   = ncfile.createVariable('Depth',np.float32,('y'))
Oxygen  = ncfile.createVariable('Dissolved oxygen' ,np.float32,('y','x'))
#time_c = ncfile.createVariable('Month',np.int,('y','x'))
#mts[:,:] = mts[:,:]
#dpt[:] = depth[:]
Oxygen[:,:]       = oxy[:,:]

print(ncfile)
ncfile.close(); print('Dataset is closed')

import matplotlib as mpl
from matplotlib import pyplot as plt

ds = xr.open_dataset('NORDIC_3902106_oxy.nc')
mod_ox = ds.ergom_t_o2.values
mod_dp = ds.deptht.values
mod_lat = ds.nav_lat.values
mod_lon = ds.nav_lon.values
print(np.shape(mod_ox))
#mod_dp = np.tile(mod_dp,(63,1))

plt.figure()
pcol = plt.pcolormesh(np.linspace(0,186,63),-mod_dp,1000*mod_ox,cmap='coolwarm')
plt.clim(0,0.4)
plt.colorbar()
plt.ylim(-200,0)
plt.savefig('ergom_oxy.png')

plt.figure(figsize=(30,40))
pcol = plt.scatter(oxy,-depth,c=(178/255,223/255,138/255),s=100)
pcol = plt.scatter(1000*mod_ox,np.transpose(np.tile(-mod_dp,(63,1))),c=(31/255,120/255,180/255),s=100)
plt.ylabel('Depth (m)',fontsize=60)
plt.yticks(fontsize=60)
plt.xlabel('Dissolved oxygen (mol m⁻³)',fontsize=60)
plt.xticks(fontsize=60)
plt.grid()
plt.title('Jan-Apr 2020',fontsize=80)
plt.savefig('polish_oxy.png')

print(np.shape(mod_dp))
print(np.linspace(0,180,61))
plt.figure()
for i in range(0,179):
    plt.scatter(np.tile(i,162),-depth[:,i],c=oxy[:,i],cmap='coolwarm')
plt.ylim(-200,0)
plt.clim(0,0.4)
plt.xticks([0,50,100,150],[dates[0,0],dates[0,50],dates[0,100],dates[0,150]])
plt.colorbar()
plt.savefig('polish_float_oxy.png')

# read in mask
ds = xr.open_dataset('NORDIC_1d_20200101_20200101_ptrc_T_0101-0101.nc')
ref_lat = ds.nav_lat.values
ref_lon = ds.nav_lon.values
ref_ox  = ds.ergom_t_o2.values
ref_ox[ref_ox>0] = 0

#ref_dep = 0
print('ref depth')
mod_ind = 10
print(mod_dp[0])
real_dep = mod_dp[mod_ind]
print(real_dep)
depth = depth - real_dep

oxy_lev=np.zeros((179))

for i in range(0,179):
    depth_now = np.abs(np.squeeze(depth[:,i]))
    depth_now = depth_now[np.isfinite(depth_now)]
    if (np.nanmax(depth[:,i])>0):
        ref_dep=np.int(np.argmin(depth_now))
        oxy_lev[i]=oxy[ref_dep,i]

print(np.shape(ref_ox))
plt.figure()
plt.pcolormesh(ref_lon,ref_lat,np.squeeze(ref_ox[0,0,:,:]),cmap='cmo.rain')
plt.scatter(long[ref_dep,:],lat[ref_dep,:],c=oxy_lev[:],cmap='cmo.thermal')
plt.colorbar()
plt.clim(0,0.44)
plt.ylim(54,59)
plt.xlim(16,22)
plt.title('Float dissolved oxygen at ~{}m (mol m⁻³)'.format(round(real_dep)))
plt.savefig('float_trajectory_{}.png'.format(round(real_dep)))

plt.figure()
plt.pcolormesh(ref_lon,ref_lat,np.squeeze(ref_ox[0,0,:,:]),cmap='cmo.rain')
plt.scatter(mod_lon[:],mod_lat[:],c=1000*mod_ox[mod_ind,:],cmap='cmo.thermal')
plt.colorbar()
plt.clim(0,0.44)
plt.ylim(54,59)
plt.xlim(16,22)
plt.title('ERGOM dissolved oxygen at {}m (mol m⁻³)'.format(round(real_dep)))
plt.savefig('model_trajectory_{}.png'.format(round(real_dep)))

# also want to plot depth at which oxygen concentration first dips below given level...

hyp_lim = 0.01
mod_lim = np.zeros((np.size(mod_ox,1)))
for t in range(0,np.size(mod_ox,1)):
    ox_prof = np.zeros((56))
    ox_prof = ox_prof + 1000*np.squeeze(mod_ox[:,t])
    ox_prof[ox_prof==0] = np.nan
    if any(ox_prof<hyp_lim):
        dp_lim = mod_dp[ox_prof<hyp_lim]
        mod_lim[t] = np.nanmin(dp_lim)
    else:
        mod_lim[t] = np.nan

plt.figure()
plt.pcolormesh(ref_lon,ref_lat,np.squeeze(ref_ox[0,0,:,:]),cmap='cmo.rain')
plt.scatter(mod_lon[:],mod_lat[:],c=mod_lim[:],cmap='cmo.thermal',edgecolors='k',linewidths=0.5)
plt.colorbar()
plt.clim(55,110)
plt.ylim(54,59)
plt.xlim(16,22)
plt.title('ERGOM depth (m) at which oxygen < {} mol m⁻³'.format(hyp_lim))
plt.savefig('model_depth_{}.png'.format(hyp_lim))

obs_lim = np.zeros((np.size(oxy,1)))
for t in range(0,np.size(obs_lim)):
    #ox_prof = np.zeros((np.size(oxy,0)))
    #ox_prof = ox_prof + np.squeeze(oxy[:,t])
    #ox_prof[ox_prof==0] = np.nan
    ox_prof = np.squeeze(oxy[:,t])
    dp_prof = np.squeeze(depth[:,t]) + real_dep
    if any(ox_prof<hyp_lim):
        dp_lim = dp_prof[ox_prof<hyp_lim]
        obs_lim[t] = np.nanmin(dp_lim)
    else:
        obs_lim[t] = np.nan
print(obs_lim)
plt.figure()
plt.pcolormesh(ref_lon,ref_lat,np.squeeze(ref_ox[0,0,:,:]),cmap='cmo.rain')
plt.scatter(long[ref_dep,:],lat[ref_dep,:],c=obs_lim[:],cmap='cmo.thermal',edgecolors='k',linewidths=0.5)
plt.colorbar()
plt.clim(55,110)
plt.ylim(54,59)
plt.xlim(16,22)
plt.title('Float depth (m) at which oxygen < {} mol m⁻³'.format(hyp_lim))
plt.savefig('float_depth_{}.png'.format(hyp_lim))

