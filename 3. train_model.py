import numpy as np
from models import nvidia
from models import googlenet

WIDTH = 160 
HEIGHT = 90 
LR = 1e-3
EPOCHS = 200 
LR_EPOCHS_DECAY = 50 
EPOCHS_BEFORE_SAVE = 10 

BATCH_SIZE = 256 
VAL_PERC = 0.05

MODEL_NAME = "ctr-nvidia.model"
#MODEL_NAME = "ctr-googlenet.model"
PREV_MODEL = "ctr-nvidia-prev.model"
#PREV_MODEL = "ctr-googlenet-prev.model"
LOAD_MODEL = False

train_loc = "E:/Users/Megaport/Documents/Data/CTR-AI/training_data/preprocessed/training_data_merge_trainset.npy"
test_loc = "E:/Users/Megaport/Documents/Data/CTR-AI/training_data/preprocessed/training_data_merge_valset.npy"

save_loc = "E:/Users/Megaport/Documents/Data/CTR-AI/models/"

model = nvidia(WIDTH, HEIGHT, LR, output=3, model_name=MODEL_NAME)
#model = googlenet(WIDTH, HEIGHT, 3, LR, output=3, model_name=MODEL_NAME)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('### LOADED PREVIOUS MODEL ###')
          
          
train = np.load(train_loc)
test = np.load(test_loc)

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
test_y = [i[1] for i in test]


train_data = [] 

for e in range(EPOCHS):
    
    #LR decay every LR_EPOCHS_DECAY steps
    if (e%LR_EPOCHS_DECAY == 0 and e > 0):
        LR = LR / 10
#    
#    if (LR < 1e-5):
#        LR = 1e-4
    
    print("Starting epoch {}, with learning rate {}".format(e, LR))
    
    model.fit({'input': X}, {'targets': Y}, n_epoch = 1, validation_set = ({'input': test_x}, {'targets': test_y}),
                snapshot_step = 2500, show_metric = True, run_id = MODEL_NAME, 
                batch_size = BATCH_SIZE)
    
    if (e%EPOCHS_BEFORE_SAVE == 0 and e > 0):
        print("### Saving model ###")
        model.save(save_loc + MODEL_NAME+"-EPOCH_{}".format(e))

print("Epochs over, saving last version...")
model.save(save_loc + MODEL_NAME+"-EPOCH_{}".format(e))
print("Save completed - exit")