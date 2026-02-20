#uncomment above to load a solution
nbouts=len(fr)
speedTracks=np.zeros((nbouts,30))
for i in range(nbouts-1):
    if fr[i]>15 and fr[i]<len(speed)-15:
        speedTracks[i,:]=speed[fr[i]-15:fr[i]+15]

plt.imshow(speedTracks, aspect='auto',clim=(0,10),origin='lower')
plt.plot(np.nanmean(speedTracks,axis=0)*100,'k')
plt.axvline(15,ls='--',color='k')