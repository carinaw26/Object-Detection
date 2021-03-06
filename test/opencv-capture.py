# import the opencv library
import cv2
import platform
  
  
# define a video capture object
cv2.namedWindow('Video Capture')
video_index = 0
if platform.system() == 'Darwin':
    video_index = 1
vid = cv2.VideoCapture(video_index)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('Video Capture', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
