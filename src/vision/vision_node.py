import logging
from pathlib import Path

import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

logger = logging.getLogger(__name__)


def capture_and_save(n=1, filename="capture.jpg", 
  output_folder="images", camera_index=0, warmup_frames=5
  ):  
  """
  Capture n frames from the camera, save each to output_folder, 
  and return a list of frames (BGR numpy arrays).
  """  
  save_dir = Path(output_folder)
  save_dir.mkdir(parents=True, exist_ok=True)
  filepath = save_dir / filename
  
  cap = cv2.VideoCapture(camera_index)
  
  if not cap.isOpened():
    raise RuntimeError(f"Could not open camera at index {camera_index}. Check permissions or connection.")

  for _ in range(warmup_frames):
    cap.read()
  
  frames = []
  
  for i  in range(n):
    ret, frame = cap.read()
    if not ret or frame is None:
      cap.release()
      raise RuntimeError(f"Failed to grab frame {i+1}/{n} from the camera.")
    frames.append(frame)
  
  cap.release()
  return frames


def detect_face_and_save(model_path: str, filename: str, output_folder: str = "images", camera_index: int = 0):
    """Capture a frame from the camera, run TF-Lite face detection, filter outputs, and save."""
    save_dir = Path(output_folder)
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / filename
    
    # 1. Setup Model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    model_height, model_width = input_shape[1], input_shape[2]
    
    # 2. Capture Frame
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")
    
    for _ in range(5): # Warmup
        cap.read()
        
    ret, image = cap.read()
    cap.release()
    
    if not ret or image is None:
        raise RuntimeError("Failed to grab a frame from the camera.")
    
    # 3. Pre-process
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_frame = cv2.resize(rgb_frame, (model_width, model_height))
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0
    
    # 4. Infer
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # 5. Extract Raw Outputs (BlazeFace outputs 2 tensors)
    out1 = interpreter.get_tensor(output_details[0]['index'])
    out2 = interpreter.get_tensor(output_details[1]['index'])
    
    # Determine which tensor is the coordinates and which is the scores
    # Scores usually have a final dimension of 1.
    if out1.shape[-1] == 1:
        scores_tensor = out1[0]
        boxes_tensor = out2[0]
    else:
        scores_tensor = out2[0]
        boxes_tensor = out1[0]
        
    # 6. Post-Process: Convert logits to probabilities (0.0 to 1.0)
    # We use np.clip to prevent mathematical overflows in np.exp
    clipped_scores = np.clip(scores_tensor.flatten(), -10, 10)
    probabilities = 1 / (1 + np.exp(-clipped_scores))
    
    # 7. Post-Process: Filter by Confidence Threshold
    confidence_threshold = 0.70
    valid_indices = np.where(probabilities > confidence_threshold)[0]
    
    filtered_boxes = boxes_tensor[valid_indices]
    filtered_scores = probabilities[valid_indices]
    
    # 8. Post-Process: Non-Maximum Suppression (NMS)
    # The model triggers multiple overlapping boxes for a single face.
    # NMS merges them into one distinct detection per face.
    nms_boxes = []
    for box in filtered_boxes:
        # OpenCV NMS expects a standard list of [x, y, w, h] floats
        nms_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
        
    iou_threshold = 0.3
    final_indices = cv2.dnn.NMSBoxes(
        bboxes=nms_boxes, 
        scores=filtered_scores.tolist(), 
        score_threshold=confidence_threshold, 
        nms_threshold=iou_threshold
    )
    
    # Flatten the result to a simple list of distinct faces
    final_faces = final_indices.flatten().tolist() if len(final_indices) > 0 else []
    
    # 9. Save and Return
    cv2.imwrite(str(filepath), image)
    logger.info(f"Saved inference image. Detected {len(final_faces)} faces.")
      
    return image, final_faces