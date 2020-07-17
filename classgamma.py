import numpy as np
import cv2
from scipy.misc import imresize
import time
from keras.models import load_model
from grabscreen import grab_screen
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines(image):
    """ Takes in a road image, re-sizes for the model,
    predicts the lane to be drawn from the model in G color,
    recreates an RGB image of a lane and merges with the
    original road image.
    """
    #print(image.shape)
    # Get image ready for feeding into model
    small_img = imresize(image, (80, 160, 3))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(small_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)

    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)
    
    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))

    # Re-size to match the original image
    lane_image = imresize(lane_drawn, (561, 1001, 3))

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)

    return lane_image, result

def maskedLane(img,mask):
    lowerGreen = np.array([0,150,0])
    upperGreen = np.array([0,255,0])
    maskedGreen = cv2.inRange(mask,lowerGreen,upperGreen)
    maskedGreen =cv2.resize(maskedGreen,(img.shape[1],img.shape[0]))
    roi = cv2.bitwise_and(img,img,mask=maskedGreen)
    return roi

def gamma_correction_auto(RGBimage, equalizeHist = False):
    originalFile = RGBimage.copy()
    red = RGBimage[:,:,2]
    green = RGBimage[:,:,1]
    blue = RGBimage[:,:,0]

    vidsize = (600, 1000)
    forLuminance = cv2.cvtColor(originalFile,cv2.COLOR_BGR2YUV)
    Y = forLuminance[:,:,0]
    totalPix = vidsize[0]* vidsize[1]
    summ = np.sum(Y[:,:])
    Yaverage = np.divide(totalPix,summ)

    epsilon = 1.19209e-007
    correct_param = np.divide(-0.3,np.log10([Yaverage + epsilon]))
    correct_param = 0.7 - correct_param 

    red = red/255.0
    red = cv2.pow(red, correct_param)
    red = np.uint8(red*255)
    if equalizeHist:
        red = cv2.equalizeHist(red)
    
    green = green/255.0
    green = cv2.pow(green, correct_param)
    green = np.uint8(green*255)
    if equalizeHist:
        green = cv2.equalizeHist(green)
        
    blue = blue/255.0
    blue = cv2.pow(blue, correct_param)
    blue = np.uint8(blue*255)
    if equalizeHist:
        blue = cv2.equalizeHist(blue)
    
    output = cv2.merge((blue,green,red))
    #print(correct_param)
    return output

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    # Load Keras model
    model = load_model('full_CNN_model.h5')
    # Create lanes object
    lanes = Lanes()

    while True:
        last_time = time.time()

        screen = grab_screen(region=(0, 40, 1000, 600))
        capture = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB) # RGB
        gamma = gamma_correction_auto(capture, equalizeHist = False)
        output, green = road_lines(gamma)
        road = maskedLane(gamma, output)
        # gamma = gamma_correction_auto(road, equalizeHist = False)

        result = cv2.addWeighted(capture, 1, road, 1, 0)
        
        #cv2.imshow('Screen Capture', capture)
        #cv2.imshow('Ego Lane', green)
        cv2.imshow('Class wise Gamma', result)

        print("fps: {}".format(1 / (time.time() - last_time)))
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
