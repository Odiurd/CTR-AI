import numpy as np
import cv2
import sys

path = "E:/Users/Megaport/Documents/Data/CTR-AI/training_data/to_be_validated/"

# Name format: yyyy-mm-dd_hh-mm-ss_user_track_mode_code. E.g. 2018-08-05_17-10-50_N_coco-park_N_0
def main():
    file = str(sys.argv[1]) + ".npy"
    train_data = np.load(path + file)
    
    for data in train_data:
        img = data[0]
        choice = data[1]
        
        cv2.imshow(file,img)
        print(choice)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()