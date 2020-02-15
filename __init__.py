import cv2
import numpy as np
from time import sleep, time

def initialize(video: str) -> tuple:
    _capture = cv2.VideoCapture(video) #initialize the video.
    _subtractor = cv2.bgsegm.createBackgroundSubtractorMOG() #removes background from moving frames.
    return _capture, _subtractor

#get contours of the current frame
def get_contours(frame):
    return cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#treats the quality of the frame obtained in real time.
def treat_frame(frame, subtractor):
    _gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #apply the grayscale filter.
    _blur = cv2.GaussianBlur(_gray, (3, 3), 5) #removes imperfections from the image, making the image blurry.
    return subtractor.apply(_blur) #apply image subtraction.

#control matrix
def get_kernel(dilate, _ksize=(5,5)):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, _ksize)

# This method hides holes inside the images, 
# for example: a white car with black dots, the 
# black dots would be mixed with the background. 
# Once this method is applied, a separation is performed
# It also improves the quality and size of the frame.
def expand_frame(frame, _number_repetitions=5):
    _dilate_frame = cv2.dilate(frame, np.ones((5, 5)))
    _karnel = get_kernel(_dilate_frame)

    for i in range(_number_repetitions):
        _dilate_frame = cv2.morphologyEx(_dilate_frame, cv2.MORPH_CLOSE, _karnel)
    
    return _dilate_frame

#get the center of the frame.
def get_centroid(x, y, w, h) -> tuple:
    return x + int(w / 2), y + int(h / 2)

#vehicle crossing control line
def draw_marking_line(frame, line_position, _color=(255, 127, 0)):
    cv2.line(frame, (25, _line_position), (1200, _line_position), _color, 3)

#draws a rectangle around the vehicle located
def mark_vehicles(frame, x, y, w, h):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    _center = get_centroid(x, y, w, h)
    cv2.circle(frame, _center, 4, (0, 0, 255), -1)
    return _center

#checks if the vehicle has passed over the motion control line
def validate_detections(frame, detected, line_position, number_vehicles, _accuracy=6):
    for (x, y) in detected:
        # print(f'y = {y} | s = {line_position + _accuracy}')
        if y < (line_position + _accuracy) and y > (line_position - _accuracy):
            number_vehicles += 1
            draw_marking_line(frame, line_position, _color=(0, 127, 255))
            detected.remove((x, y))
            print(f'number of cars detected at the moment: {_number_vehicles}')

    return detected, number_vehicles

def imshow(frame, number_vehicles, dilate):
    cv2.putText(frame, f'Vehicles: {number_vehicles}', (450, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow('Original video with detection', frame)
    cv2.imshow('Expanded video', dilate)

if __name__ == "__main__":
    _minimum_dimensions = [80, 80] #minimum dimensions of the vehicle.
    _fps_video = 60 #Frames Per Second.
    _path_video = 'demo_video/traffic.mp4' #vehicle traffic demonstration directory.
    _process = True #loop control variable.
    _detected = [] #vehicle monitoring list
    _line_position = 550 #motion control line
    _number_vehicles = 0

    _capture, _subtractor = initialize(video=_path_video)

    while(_process):
        _, _frame = _capture.read() #get the frames.
        sleep(float(1 / _fps_video)) #controls the number of frames per second so that you don't find processing executable.

        _subtracted_frame = treat_frame(_frame, _subtractor)
        _dilate_frame = expand_frame(_subtracted_frame)
        _contours, _ = get_contours(_dilate_frame)

        draw_marking_line(_frame, _line_position)

        for (_, contour) in enumerate(_contours):
            (x, y, w, h) = cv2.boundingRect(contour) #take the dimensions
            
            if not((w >= _minimum_dimensions[0]) and (h >= _minimum_dimensions[1])):
                continue
            
            _vehicle_detected = mark_vehicles(_frame, x, y, w, h)
            _detected.append(_vehicle_detected)
            _detected, _number_vehicles = validate_detections(_frame, _detected, _line_position, _number_vehicles)
        
        imshow(_frame, _number_vehicles, _dilate_frame)

        if cv2.waitKey(1) == 27:
            _process = False

cv2.destroyAllWindows()
_capture.release()
        


