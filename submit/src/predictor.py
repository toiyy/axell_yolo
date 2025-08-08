import cv2
import torch
import numpy as np
# Import the necessary post-processing function
from ultralytics.utils.ops import non_max_suppression

class Predictor(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method"""
        try:
            model_file = f"{model_path}/best.pt"
            checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
            cls.model = checkpoint['model']
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model.to(cls.device)
            cls.model.eval()
            cls.model.float()
            cls.img_size = 640
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True

    @classmethod
    def predict(cls, input):
        """Predict method"""
        # --- Preprocessing ---
        orig_shape = input.shape
        img_resized = cv2.resize(input, (cls.img_size, cls.img_size), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(cls.device)

        # --- Inference ---
        with torch.no_grad():
            # The model returns a raw tensor or a list of tensors
            raw_results = cls.model(tensor)

        # --- Postprocessing with NMS ---
        # The output might be a tuple (e.g., (prediction, other_stuff)), get the main prediction tensor.
        # This is common in many model architectures.
        prediction_tensor = raw_results[0] if isinstance(raw_results, tuple) else raw_results
        
        # Apply Non-Maximum Suppression
        # conf_thres: confidence threshold
        # iou_thres: IoU threshold for NMS
        detections = non_max_suppression(prediction_tensor, conf_thres=0.01, iou_thres=0.45, classes=None, agnostic=False, max_det=5)
        
        predict_list = []
        
        # Detections is a list of tensors, one for each image in the batch.
        # We have a batch of 1, so we take detections[0].
        final_detections = detections[0]

        if final_detections is not None and len(final_detections) > 0:
            # Move detections to CPU
            final_detections = final_detections.cpu()

            # Sort by score in descending order
            final_detections = final_detections[final_detections[:, 4].argsort(descending=True)]

            # Take only the top 5 detections
            final_detections = final_detections[:5]

            # Scale coordinates back from the resized image (cls.img_size) to the original image size.
            scale_w = orig_shape[1] / cls.img_size
            scale_h = orig_shape[0] / cls.img_size
            
            for det in final_detections:
                x1, y1, x2, y2, score, label_id = det
                
                # Apply scaling to each coordinate
                x1_orig = x1 * scale_w
                y1_orig = y1 * scale_h
                x2_orig = x2 * scale_w
                y2_orig = y2 * scale_h

                # Convert to the required [x, y, width, height] format.
                bbox = [x1_orig.item(), y1_orig.item(), (x2_orig - x1_orig).item(), (y2_orig - y1_orig).item()]

                # Competition expects 1-indexed labels
                label = int(label_id.item()) + 1

                predict_list.append({
                    "category_id": label,
                    "bbox": bbox,
                    "score": score.item()
                })
                
        return predict_list
