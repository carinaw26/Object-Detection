# Specify model imports
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
import os
import platform
import tensorflow as tf
from flask import Response
from flask import request
from flask import Flask
from flask import render_template
import threading
import argparse

print ("Disable GPU if necessary")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Create object detector
class TFObjectDetector():
  
  # Constructor
  def __init__(self, path_to_object_detection = './models/research/object_detection/configs/tf2',\
    path_to_model_checkpoint = './checkpoint', path_to_labels = './labels.pbtxt',\
      model_name = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'):
    self.model_name = model_name
    self.pipeline_config_path = path_to_object_detection
    self.pipeline_config = os.path.join(f'{self.pipeline_config_path}/{self.model_name}.config')
    self.full_config = config_util.get_configs_from_pipeline_file(self.pipeline_config)
    self.path_to_model_checkpoint = path_to_model_checkpoint
    self.path_to_labels = path_to_labels
    self.setup_model()


  # Set up model for usage
  def setup_model(self):
    self.build_model()
    self.restore_checkpoint()
    self.detection_function = self.get_model_detection_function()
    self.prepare_labels()

  # Build detection model
  def build_model(self):
    model_config = self.full_config['model']
    assert model_config is not None
    self.model = model_builder.build(model_config=model_config, is_training=False)
    return self.model

  # Restore checkpoint into model
  def restore_checkpoint(self):
    assert self.model is not None
    self.checkpoint = tf.train.Checkpoint(model=self.model)
    self.checkpoint.restore(os.path.join(self.path_to_model_checkpoint, 'ckpt-0')).expect_partial()
    
  # Get a tf.function for detection
  def get_model_detection_function(self):
    assert self.model is not None
    
    @tf.function
    def detection_function(image):
      image, shapes = self.model.preprocess(image)
      prediction_dict = self.model.predict(image, shapes)
      detections = self.model.postprocess(prediction_dict, shapes)
      return detections, prediction_dict, tf.reshape(shapes, [-1])
    
    return detection_function
    
  # Prepare labels
  # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
  def prepare_labels(self):
    label_map = label_map_util.load_labelmap(self.path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map,
        max_num_classes=label_map_util.get_max_label_map_index(label_map),
        use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    self.label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
    
  # Get keypoint tuples
  # Source: https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/inference_tf2_colab.ipynb
  def get_keypoint_tuples(self, eval_config):
    tuple_list = []
    kp_list = eval_config.keypoint_edge
    for edge in kp_list:
      tuple_list.append((edge.start, edge.end))
    return tuple_list

  # Prepare image
  def prepare_image(self, image):
    return tf.convert_to_tensor(
      np.expand_dims(image, 0), dtype=tf.float32
    )
    
  # Perform detection
  def detect(self, image, label_offset = 1):
    # Ensure that we have a detection function
    assert self.detection_function is not None
    
    # Prepare image and perform prediction
    image = image.copy()
    image_tensor = self.prepare_image(image)
    detections, predictions_dict, shapes = self.detection_function(image_tensor)

    # Use keypoints if provided
    keypoints, keypoint_scores = None, None
    if 'detection_keypoints' in detections:
      keypoints = detections['detection_keypoints'][0].numpy()
      keypoint_scores = detections['detection_keypoint_scores'][0].numpy()
    
    # Perform visualization on output image/frame 
    viz_utils.visualize_boxes_and_labels_on_image_array(
      image,
      detections['detection_boxes'][0].numpy(),
      (detections['detection_classes'][0].numpy() + label_offset).astype(int),
      detections['detection_scores'][0].numpy(),
      self.category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=25,
      min_score_thresh=.40,
      agnostic_mode=False,
      keypoints=keypoints,
      keypoint_scores=keypoint_scores,
      keypoint_edges=self.get_keypoint_tuples(self.full_config['eval_config']))
    
    # Return the image
    return image

  # Predict video from capture
  def detect_live_video(self):
    global OBJECT_DETECTION_LIVE, lock, outputFrame

    # define a video capture object
    # cv2.namedWindow('frame')
    video_index = 0
    if platform.system() == 'Darwin':
      video_index = 1
    vid = cv2.VideoCapture(video_index)
  
    while(OBJECT_DETECTION_LIVE): 
      # Capture the video frame
      # by frame
      ret, frame = vid.read()

      # Perform object detection
      frame_out = self.detect(frame)
      
      with lock:
        outputFrame = frame_out.copy()

      # Ignore following. Will use self.OBJECT_DETECTION_LIVE
      # the 'q' button is set as the
      # quitting button you may use any
      # desired button of your choice
      #if cv2.waitKey(1) & 0xFF == ord('q'):
      #  break


OBJECT_DETECTION_LIVE = True
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

def shutdown_server():
  func = request.environ.get('werkzeug.server.shutdown')
  if func is None:
    raise RuntimeError('Not running with the Werkzeug Server')
  func()
    
@app.get('/shutdown')
def shutdown():
  global OBJECT_DETECTION_LIVE
  OBJECT_DETECTION_LIVE = False
  shutdown_server()
  return 'Server shutting down...'

# Feed video through Web
def render_video_to_web():
  global lock, outputFrame
  while(True): 
    with lock:
      if outputFrame is None:
        continue
      # encode the frame in JPEG format
      (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
      if not flag:
        continue
		  
    # yield the output frame in the byte format
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
      bytearray(encodedImage) + b'\r\n')
            
            
@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")  

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(render_video_to_web(),
	  mimetype = "multipart/x-mixed-replace; boundary=frame")
    
    
if __name__ == '__main__':
  # python ah_object_detection_web.py --ip 0.0.0.0 --port 8000
  # construct the argument parser and parse command line arguments
  ap = argparse.ArgumentParser()
  ap.add_argument("-i", "--ip", type=str, required=True,
    help="ip address of the device")
  ap.add_argument("-o", "--port", type=int, required=True,
    help="ephemeral port number of the server (1024 to 65535)")
  args = vars(ap.parse_args())
  
  ot_model = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"
  #print("Run Object Detection ...")
  TF_WORK_DIR="../../workspace/Tensorflow"
  detector = TFObjectDetector(TF_WORK_DIR + '/models/research/object_detection/configs/tf2', 
        TF_WORK_DIR + '/workspace/pre-trained-models/'+ ot_model +'/checkpoint', 
        TF_WORK_DIR + '/models/research/object_detection/data/mscoco_label_map.pbtxt', 
        ot_model)
  #print("Detect still image ...")
  #detector.detect_image('c:/tmp/beach.png', 'c:/tmp/beachout.png')
  #print("Detect viedo ...")
  #detector.detect_video('c:/tmp/AHD_30s_SD360P.mp4', 'c:/tmp/AHD_30s_SD360P_out.mp4')
  
  t = threading.Thread(target=detector.detect_live_video, args=())
  t.daemon = True
  t.start()
  
  # start the flask app
  app.run(host=args["ip"], port=args["port"], debug=True,
    threaded=True, use_reloader=False)