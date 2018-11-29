
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from tqdm import tqdm

params = (0.7841752015528167, 2.1022996225901887, 0.575376571272233, 0.08088203534323207)
# get just one sample from vote per money distribution
x = []
bad_samples = 0
for i in tqdm(range(0, 10000)):
	sample = st.nct.rvs(loc=params[-2], scale=params[-1], *params[:-2], size=1)[0]
	if(sample > 10):
		bad_samples+=1
		continue
	x.append(sample)

print(bad_samples)
plt.hist(x, bins = 100)
plt.show()