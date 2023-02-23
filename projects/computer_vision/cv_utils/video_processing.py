import cv2

def display_video(file_path_name):
    # Create video capture object and read from input file
    capture = cv2.VideoCapture(file_path_name)
    # check if camer opened successfully
    if(capture.isOpened() == False):
        print("Error Opening video stream or file")

    # Read until video is completed
    while(capture.isOpened()):
        # capture frame by frame
        ret, frame = capture.read()
        if ret == True:
            # display the resulting frame
            cv2.imshow('Frame',frame)
            # press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        
        # Break the loop
        else:
            break
    
    # When video is complete release the capture object
    capture.release()

    # Close all frames
    cv2.destroyAllWindows()