# -*- coding: UTF-8 -*-
#提取quit ratio文件，生成CDF
import random
import string
import matplotlib
import numpy
import scipy
import pyparsing
import math
import matplotlib.pyplot as plt
from matplotlib import pylab
from pylab import *
import random


plt.figure(figsize=(14, 10))
##################QoE#####################

plt.subplot(3, 2, 3)

s_5 = [1.25]
s_4 = [6.0]
s_3 = [12.5]
s_2 = [13.6]
s_1 = [8.41]

plt.grid(True)
a = plt.bar([1], s_5, width=0.4, yerr=4.6, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='deepskyblue')
b = plt.bar([2], s_4, width=0.4, yerr=4.1, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='deepskyblue')
c = plt.bar([3], s_3, width=0.4, yerr=4.1, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='deepskyblue')
d = plt.bar([4], s_2, width=0.4, yerr=4.1, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='deepskyblue')
d = plt.bar([5], s_1, width=0.4, yerr=4.7, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='deepskyblue')



plt.xticks(range(1,6), ('5s','4s','3s','2s','1s'), fontsize =24)
plt.yticks([0,10,20],fontsize = 24)
plt.ylabel("QoE improve (%)",fontsize=24)
plt.xlabel("Prefetch length limit (sec)",fontsize=24)

plt.axis([0.5,5.5,0,20])







#############bitrate and rebuf#################

plt.subplot(3, 2, 4)


s_5 = [6.13]
s_4 = [11.5]
s_3 = [14.6]
s_2 = [23.2]
s_1 = [29.7]

plt.grid(True)
a = plt.bar([1], s_5, width=0.4, yerr=0.8, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='tomato')
b = plt.bar([2], s_4, width=0.4, yerr=1.2, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='tomato')
c = plt.bar([3], s_3, width=0.4, yerr=1.5, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='tomato')
d = plt.bar([4], s_2, width=0.4, yerr=1.8, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='tomato')
d = plt.bar([5], s_1, width=0.4, yerr=2.0, error_kw={"elinewidth":2,"ecolor":"black","capsize":6}, color='tomato')



plt.xticks(range(1,6), ('5s','4s','3s','2s','1s'), fontsize =24)
plt.yticks([0,15,30],fontsize = 24)
plt.ylabel("Wastage (%)",fontsize=24)
plt.xlabel("Prefetch length limit (sec)",fontsize=24)

plt.axis([0.5,5.5,0,30])



#plt.legend(bbox_to_anchor=(0.5,1.0),loc='lower center',prop={'size':22}, ncol=2)
plt.subplots_adjust(wspace =0.3, hspace =0)
plt.savefig('wa.svg', dpi=600, orientation='portrait', format='svg',transparent=False, bbox_inches='tight', pad_inches=0.1)
plt.show()
