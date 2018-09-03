import numpy as np
from grabscreen import grab_screen
from getkeys import key_check
import cv2
import time
import sys
import datetime

# Current emulator keyboard settings:
# J = left
# Z = forward
# L = right


def keys_to_output(keys):
    output = [0,0,0]     # [J,Z,L]
    if 'J' in keys:
        output[0] = 1
    elif 'L' in keys:
        output[2] = 1
    elif 'Z' in keys:
        output[1] = 1
        
    return output

def main(): 
    user = str(sys.argv[1]) # Distinguish among multiple users. D / N
    track = str(sys.argv[2]) # Defines track
    mode = str(sys.argv[3]) # Label collection type. A = adjustments, N = normal, D = dark
    
    FPS = 10
    FILE_SIZE = 500
    
    #File names
    path = "E:/Users/Megaport/Documents/Data/CTR-AI/training_data/to_be_validated/"
    tst = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = tst + "_" + user + "_" + track + "_" + mode
    
    
    starting_value = 0
    paused = False
    training_data = []
    
    
    #Delay while moving to emulator window    
    for i in range(1, 4):
        print(i)
        time.sleep(1)
    print('START')
    
    
    last_time = time.time()
    while(True):
        if not paused:
            # Find a way to get exact coordinates automatically
            # 55 px: window name
            screen = grab_screen(region=(0,55,645,530))
            screen = cv2.resize(screen, (160,90))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            
            #Collect data every N frames. Discard recordings without actions
            if time.time() > last_time + (1 / FPS):
                keys = key_check()       
                output = keys_to_output(keys)
                if output != [0,0,0]:
                    training_data.append([screen,output])
                    last_time = time.time() # consider moving this outside if statement
            
            #Track progress. Save subfiles if above FILE_SIZE threshold.
            if len(training_data) % 100 == 0:
                print("Current data length: ", len(training_data))
                if len(training_data) >= FILE_SIZE:
                    np.save(getFullFileName(path,file_name,starting_value), training_data)
                    print("File " + getFullFileName("",file_name,starting_value) +  "ready")
                    training_data = []
                    starting_value += 1
            
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print("Restart in 1 second")
                time.sleep(1)
                print("RESTARTED")
            else:
                print("Pause")
                paused= True
                time.sleep(1) # avoids multi press
        
        if 'P' in keys:
            print("Exit")
            break
        
        if 'O' in keys:
            print("Saving file " + file_name)
            np.save(getFullFileName(path,file_name,starting_value), training_data)
            print("Exit")
            break
    
#        cv2.imshow('window',screen)
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break
            
    # TODO: creare una lista output che elenchi i file creati, magari col comando pronto per visualizzarli
    
    
def getFullFileName(path, name, index):
    return path + name + "_" + str(index) + ".npy"


if __name__=="__main__":
    main()