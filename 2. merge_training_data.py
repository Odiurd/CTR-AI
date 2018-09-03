import numpy as np
from os import listdir
from os.path import isfile, join

loc_read = "E:/Users/Megaport/Documents/Data/CTR-AI/training_data/validated/"
loc_save = "E:/Users/Megaport/Documents/Data/CTR-AI/training_data/preprocessed/"
onlyfiles = [f for f in listdir(loc_read) if isfile(join(loc_read, f))]
loop = 0

for file in onlyfiles:
    if loop == 0:
        train_data_tot = np.load(loc_read+file)
        loop += 1
    else:    
        train_data = np.load(loc_read+file)
        train_data_tot = np.vstack((train_data_tot, train_data))

print("Total datapoints: {}".format(len(train_data_tot)))
np.save(loc_save+"training_data_merge.npy", train_data_tot)
