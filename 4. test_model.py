import argparse
import numpy as np
from grabscreen import grab_screen
from getkeys import key_check
import cv2
import time
from directkeys import PressKey, ReleaseKey, J, L, Z
from models import googlenet, nvidia


z = [0,1,0] #forward
j = [1,0,0] #left
l = [0,0,1] #right

W_RES = 160
H_RES = 90
COL_RES = 3


def keys_to_output(keys):
    # [J,Z,L]
    output = [0,0,0]
    if 'J' in keys:
        output[0] = 1
    elif 'L' in keys:
        output[2] = 1
    else:
        output[1] = 1
    return output


def left():
    PressKey(Z)
    PressKey(J)
    ReleaseKey(L)

def right():
    PressKey(Z)
    PressKey(L)
    ReleaseKey(J)
    
def forward():
    PressKey(Z)
    ReleaseKey(J)
    ReleaseKey(L)
    
    
    
def makePrediction(model, screen_res, weighted_rng=True, debug_mode=True):
    prediction = model.predict([screen_res.reshape(W_RES,H_RES,COL_RES)])[0]
    prediction = np.array(prediction) * np.array([1,1,1]) # fixes approximation errors
    
    if weighted_rng:        
        tot = np.sum(prediction)
        prediction_adj = prediction
        prediction_adj[0] = prediction[0] / tot
        prediction_adj[1] = prediction[1] / tot
        prediction_adj[2] = 1 - prediction_adj[0] - prediction_adj[1]
    
        choices = [0, 1, 2]
        prediction_choice = np.random.choice(choices, 1, p=prediction_adj.tolist())[0]
    else:
        prediction_choice = np.argmax(prediction)
        
    if debug_mode == True:
        print(prediction)

    return prediction_choice


def applyChoice(prediction_choice):
    if prediction_choice == 0:
        left()
    elif prediction_choice == 1:
        forward()
    elif prediction_choice == 2:
        right()
    
    
def main():     
    parser = argparse.ArgumentParser()
    parser.add_argument("-m",type=str, default="models/ctr-googlenet.model-EPOCH_199", help="Model name (and location)")
    #parser.add_argument("-m",type=str, default="models/ctr-nvidia.model-EPOCH_199", help="Model name (and location)")
    parser.add_argument("-deb",action="store_true", default=False, help="Activates debug mode: prints choices made by the CNN model")
    parser.add_argument("-dis",action="store_true", default=False, help="Creates a cv2 window displaying the game")
    
    args = parser.parse_args()
    MODEL_NAME = args.m
    debug_mode = args.deb
    display_mode = args.dis
    
    model = googlenet(W_RES, H_RES, 3, lr=0.0001, output=3)
    #model = nvidia(W_RES, H_RES, lr=0.0001, output=3)
    model.load(MODEL_NAME)
    
    weighted_rng = True  # If False applies argmax to find where to steer 
    
    #Delay while moving to emulator window    
    for i in range(1, 4):
        print(i)
        time.sleep(1)
    print('START')

    paused = False
    print('START')
    
    while(True):
        if not paused:
            # Find a way to get exact coordinates automatically
            # 55 px accounts for title bar. 
            screen = grab_screen(region=(0,55,645,530))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            screen_res = cv2.resize(screen, (W_RES,H_RES))
            
            # CNN model
            prediction_choice = makePrediction(model, screen_res, weighted_rng, debug_mode)
            applyChoice(prediction_choice)
            
            
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
                ReleaseKey(L)
                ReleaseKey(J)
                ReleaseKey(Z)
                time.sleep(1)
        
        if 'P' in keys:
            break

        if display_mode == True:
            cv2.imshow('window',screen)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break



if __name__== "__main__":
    main()
