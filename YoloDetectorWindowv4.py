import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import os

# Little function to resize in keeping the format ratio
# Cf. https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    image = image
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def set_model_and_treat_image(_YOLO_CONFIG_FILE,_YOLO_WEIGHTS_FILE,_COCO_LABELS_FILE,  IMAGE_model):
  # initialiser le reseau Darknet: Verifier bien le chemin du fichier cfg et weights
  yolo = cv2.dnn.readNetFromDarknet(_YOLO_CONFIG_FILE, _YOLO_WEIGHTS_FILE)

  # Selectionner la derniere couche
  yololayers = [yolo.getLayerNames()[i -1] for i in yolo.getUnconnectedOutLayers()]

  # Convertir l'image en blob
  blobimage = cv2.dnn.blobFromImage(IMAGE_model, 1 / 255.0, (416, 416), swapRB=True, crop=False)
  # Insérer l'image
  yolo.setInput(blobimage)

  # Traiter l'image
  layerOutputs = yolo.forward(yololayers)

  with open(_COCO_LABELS_FILE, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
  print(labels)
  np.random.seed(45)
  BOX_COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

  return layerOutputs, BOX_COLORS, labels


def output_results(layerOutputs,labels,W,H,CONFIDENCE_MIN = 0.1,unique_box = True):
  # Récupération des résultats
  boxes_detected = []
  confidences_scores = []
  labels_detected = []
  label_names = []
  # loop over each of the layer outputs
  for output in layerOutputs:
    # loop over each of the detections
    for detection in output:
      # extract the class ID and confidence (i.e., probability) of the current object detection
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]
      # Take only predictions with confidence more than CONFIDENCE_MIN thresold
      if confidence > CONFIDENCE_MIN:
        # Bounding box
        # box = detection[0:4] * np.array([W, H, W, H])
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")

        # Use the center (x, y)-coordinates to derive the top and left corner of the bounding box
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))

        # update our result list (detection)
        boxes_detected.append([x, y, int(width), int(height)])
        confidences_scores.append(float(confidence))
        labels_detected.append(classID)


        label_names = [labels[i] for i in labels_detected]

  # Remove unnecessary boxes
  if unique_box:
    final_boxes = cv2.dnn.NMSBoxes(boxes_detected, confidences_scores, 0.35, 0.35)
    boxes_detected, confidences_scores, labels_detected = select_box_tool(final_boxes,
                                                                          boxes_detected,
                                                                          confidences_scores,
                                                                          labels_detected)

  return boxes_detected, confidences_scores, labels_detected, label_names

def plotting_results(boxes_detected, confidences_scores, labels_detected,BOX_COLORS, labels,IMAGE):
  # Plot rectangles
  image = IMAGE.copy()
  list_already_happend = []
  nb_results = len(labels_detected)
  if nb_results > 0:
    for i in range(nb_results):
      # Extract the bounding box coordinates
      (x, y) = (boxes_detected[i][0], boxes_detected[i][1])
      (w, h) = (boxes_detected[i][2], boxes_detected[i][3])
      # Draw a bounding box rectangle and label on the image
      color = [int(c) for c in BOX_COLORS[labels_detected[i]]]
      cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
      score = str(round(float(confidences_scores[i]) * 100, 1)) + "%"
      text = "{}: {}".format(labels[labels_detected[i]], score)
      cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

  if nb_results > 0:
    #ResizeWithAspectRatio(image, width=700).shape
    # Convertir l'image de BGR (OpenCV) à RGB (Matplotlib)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Afficher l'image avec Matplotlib
    plt.imshow(image_rgb)
    plt.axis('off')  # Masquer les axes
    plt.show()
  else:
    print("No object detected")

def select_box_tool(_final_boxes, _boxes_detected, _confidences_scores, _labels_detected):
  boxes_detected2 = []
  confidences_scores2 = []
  labels_detected2 = []

  for indice in _final_boxes:
    boxes_detected2.append(_boxes_detected[indice])
    confidences_scores2.append(_confidences_scores[indice])
    labels_detected2.append(_labels_detected[indice])

  boxes_detected_output, confidences_scores_output, labels_detected_output = boxes_detected2, confidences_scores2, labels_detected2
  return boxes_detected_output, confidences_scores_output, labels_detected_output

def detector(name_image_path):

  YOLO_CONFIG = os.getcwd()
  COCO_LABELS_FILE = YOLO_CONFIG + '\\label\\window.names'
  YOLO_CONFIG_FILE = YOLO_CONFIG + '\\configuration\\window_yolov4.cfg'
  YOLO_WEIGHTS_FILE = YOLO_CONFIG + '\\model\\window_yolov4_last.weights'

  IMAGE = cv2.imread(name_image_path)


  # CONFIDENCE_MIN = 0.5

  layerOutputs, BOX_COLORS, labels = set_model_and_treat_image(YOLO_CONFIG_FILE,YOLO_WEIGHTS_FILE,COCO_LABELS_FILE, IMAGE)
  W = IMAGE.shape[1]
  H = IMAGE.shape[0]
  boxes_detected, confidences_scores, labels_detected, label_names = output_results(layerOutputs,labels,W,H)
  plotting_results(boxes_detected, confidences_scores, labels_detected,BOX_COLORS, labels, IMAGE)