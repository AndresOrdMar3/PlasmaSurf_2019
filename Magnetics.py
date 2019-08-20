#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:24:34 2019

@author: Andres and Vlad
"""

# libraries for data loading processing and representation
from sdas.core.client.SDASClient import SDASClient
from sdas.tests.LoadSdasData import LoadSdasData
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

# variables declaration
Range=620
InitialT=0
Mag=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
times=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
CRR=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
host='baco.ipfn.ist.utl.pt'
port=8888
Client = SDASClient(host,port)
EventN=45994
Channels=['202','203','204','205','206','207','208','209','210','211','212','213']
Ro=np.zeros((Range))
Zo=np.zeros((Range))
Zk=np.zeros((12),dtype='float')
Ro=np.zeros((Range),dtype='float')
Zo=np.zeros((Range),dtype='float')
ix=np.zeros((Range),dtype='float')
iy=np.zeros((Range),dtype='float')
Ipf=np.zeros((Range,12),dtype='float')
ang=[30,60,90,120,150,180,210,240,270,300,330,360]

# pseudo inverse matrix (12,12)
Mpf_SVD=[558380.834539853,	-423041.791246495,	263327.797101461,	-117309.938342375,	18816.4612627471,	65143.6637366474,	-147360.361179166,	270329.883501855,	-445570.201224130,	698635.177912398,	-611888.015178159,	-195478.217258221,
		-351320.568694366,	232777.956679177,	-93646.2286901853,	-7185.85477672278,	91746.1258852966,	-163729.744573662,	264323.407369461,	-409465.059257766,	645104.987090318,	-649358.957708526,	-19006.1941582380,	412828.022050860,
		206676.385056415,	-75189.5417181656,	-30685.4988131662,	120793.213499530,	-186961.544921214,	268488.847546706,	-377144.931154907,	564588.447575830,	-580158.339458776,	32947.1963442002,	321316.238799084,	-292980.584837129,
		-86324.9591244337,	-33350.9238238512,	140825.803347829,	-211379.817615473,	282636.314667562,	-359278.218670154,	491319.248511717,	-464582.003935123,	-12026.5258943471,	305457.778106958,	-282383.228035577,	214192.104195711,
		5429.02329895389,	129619.343028226,	-219957.314219746,	294373.117191320,	-349598.010209449,	433289.496594810,	-340148.855366425,	-106464.221031153,	336591.971720720,	-306460.236741382,	251785.836436946,	-135925.991520779,
		75081.6721889952,	-193286.762619824,	285950.211983755,	-334004.662558612,	383346.199296199,	-219653.163919386,	-219653.163919386,	383346.199296199,	-334004.662558611,	285950.211983754,	-193286.762619822,	75081.6721889934,
		-135925.991520780,	251785.836436947,	-306460.236741383,	336591.971720720,	-106464.221031154,	-340148.855366424,	433289.496594810,	-349598.010209447,	294373.117191319,	-219957.314219744,	129619.343028224,	5429.02329895569,
		214192.104195712,	-282383.228035577,	305457.778106959,	-12026.5258943488,	-464582.003935122,	491319.248511716,	-359278.218670153,	282636.314667561,	-211379.817615472,	140825.803347829,	-33350.9238238504,	-86324.9591244341,
		-292980.584837127,	321316.238799083,	32947.1963442008,	-580158.339458776,	564588.447575829,	-377144.931154907,	268488.847546706,	-186961.544921213,	120793.213499529,	-30685.4988131648,	-75189.5417181663,	206676.385056414,
		412828.022050857,	-19006.1941582346,	-649358.957708529,	645104.987090320,	-409465.059257768,	264323.407369461,	-163729.744573662,	91746.1258852962,	-7185.85477672143,	-93646.2286901863,	232777.956679178,	-351320.568694364,
		-195478.217258216,	-611888.015178164,	698635.177912400,	-445570.201224131,	270329.883501855,	-147360.361179165,	65143.6637366468,	18816.4612627479,	-117309.938342376,	263327.797101461,	-423041.791246493,	558380.834539849,
          -436657.366162039, 674730.893462916, -458337.073904865, 275099.193084250, -134337.982502335, 42053.1138708534, 42053.1138708538, -134337.982502335, 275099.193084250, -458337.073904863, 674730.893462913, -436657.366162035]
Mpf_SVD=np.reshape(Mpf_SVD,[12,12])

# access to the magnetic data
for i in range (0,12):
    Mag[i], times[i]=LoadSdasData(Client,'MARTE_NODE_IVO3.DataCollection.Channel_'+Channels[i],EventN-1)
Mag=np.array(Mag)
MagT=np.transpose(Mag)

# access to the r and z position
[zpos, timesZpos]=LoadSdasData(Client,'MARTE_NODE_IVO3.DataCollection.Channel_102',45994)
[rpos, timesRpos]=LoadSdasData(Client,'MARTE_NODE_IVO3.DataCollection.Channel_101',45994)

# magnetic data representation
for x in range (0,len(Mag)):
    plt.plot(times[x][InitialT:InitialT+Range]/1000,Mag[x][InitialT:InitialT+Range])
plt.title('Mirnov Data')
plt.ylabel('Magnetic Flux [Wb]')
plt.xlabel('t [ms]')
plt.show()

# z computing for each filament
for xx in range (0,12):
    Zk[xx]=0.055*np.sin(ang[xx]*(np.pi/180))

# current computing of each filament
# r position and z position computing of main plasma current
for y in range (0,Range):
    Ipf[y]=np.abs(np.dot(Mpf_SVD,MagT[y]))
    Ro[y]=np.sqrt(sum(Ipf[y]*(0.055**2)))/sum(Ipf[y])
    Zo[y]=sum(Ipf[y]*Zk)/sum(Ipf[y])

# data representation
Ipft=np.transpose(Ipf)

# Current of each of the 12 filaments
for x in range (0,len(Ipft)):
    plt.plot(times[x][InitialT:InitialT+Range]/1000,Ipft[x])
plt.title('Ip,f for each filament')
plt.ylabel('I [A]')
plt.xlabel('t [ms]')
plt.show()

# Radial position of the plasma current centroid
plt.subplot(1,2,1)
plt.plot(times[0][InitialT:InitialT+Range]/1000,Ro,label='Magentics')
plt.plot(timesRpos[InitialT:InitialT+Range]/1000,rpos[InitialT:InitialT+Range],label='MARTE')
plt.title('Radial Position')
plt.ylabel('r [m]')
plt.xlabel('t [ms]')
plt.legend(loc='upper right')
plt.grid(True)

# Z position of the plasma current centroid
plt.subplot(1,2,2)
plt.plot(times[0][InitialT:InitialT+Range]/1000,Zo,label='Magnetics')
plt.plot(timesZpos[InitialT:InitialT+Range]/1000,zpos[InitialT:InitialT+Range],label='MARTE')
plt.title('Z position')
plt.ylabel('z [m]')
plt.xlabel('t [ms]')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Animation
xTicks=np.linspace(36,56,9)
fig = plt.figure(figsize=(10,10))
ax = plt.axes(xlim=(36, 56), ylim=(-10, 10))
ax.set_xticks(xTicks, minor=False)
ax.xaxis.grid(True, which='major')
ax.yaxis.grid()
ax.set_xlabel('Radial Posotion [cm]')
ax.set_ylabel('Z Posotion [cm]')
line, = ax.plot([], [], 'bo', label='Plasma Current Centroid', linewidth=2)
circ, = ax.plot([], [], 'k', label='ISTTOK Wall', linewidth=2)
centerx, = ax.plot([], [], 'gray')
centery, = ax.plot([], [], 'gray')
ax.legend()
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    circle=np.linspace(0,6.28,100)
    yy=9.35*np.sin(circle)
    xx=9.35*np.cos(circle)+46
    line.set_data([],[])
    circ.set_data(xx,yy)
    centerx.set_data([46,46],[-1,1])
    centery.set_data([45,47],[0,0])
    time_text.set_text('')
    return line, circ, centerx, centery, time_text

def animate(i):
    r=(Ro[i]*100)+46
    z=Zo[i]*100
    line.set_data(r,z)
    time_text.set_text('time = ' + str(times[0][i]/1000) + ' ms')
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(Ro), interval=10, blit=True)
anim.save('Main Plasma Current Position.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
plt.show()