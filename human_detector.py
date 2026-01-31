import cv2
import numpy as np
import argparse
import os
import urllib.request

class ModernHumanDetector:
    def __init__(self):
        self.face_net = None
        self.person_net = None
        self.load_models()
    
    def download_model(self, url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")
    
    def load_models(self):
        print("Loading deep learning models...")
        
        # Face detection model (ResNet-based)
        face_proto = "deploy.prototxt"
        face_model = "res10_300x300_ssd_iter_140000.caffemodel"
        
        face_proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        face_model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        try:
            self.download_model(face_proto_url, face_proto)
            self.download_model(face_model_url, face_model)
            self.face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)
            print("Face detection model loaded")
        except Exception as e:
            print(f"Error loading face model: {e}")
        
        # Person detection using MobileNet-SSD
        person_proto = "MobileNetSSD_deploy.prototxt"
        person_model = "MobileNetSSD_deploy.caffemodel"
        
        person_proto_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        person_model_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
        
        try:
            if not os.path.exists(person_proto):
                print(f"Downloading {person_proto}...")
                urllib.request.urlretrieve(person_proto_url, person_proto)
            
            if not os.path.exists(person_model):
                print(f"Note: {person_model} needs manual download from:")
                print("https://github.com/chuanqi305/MobileNet-SSD")
                print("Using HOG detector as fallback...")
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            else:
                self.person_net = cv2.dnn.readNetFromCaffe(person_proto, person_model)
                print("Person detection model loaded")
        except Exception as e:
            print(f"Using HOG detector: {e}")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect_faces(self, image, confidence_threshold=0.5):
        if self.face_net is None:
            return []
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 
            1.0, 
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                faces.append({
                    'bbox': (x, y, x2 - x, y2 - y),
                    'confidence': float(confidence),
                    'type': 'face'
                })
        
        return faces
    
    def detect_persons_hog(self, image):
        boxes, weights = self.hog.detectMultiScale(
            image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        
        if len(boxes) == 0:
            return []
        
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            weights.tolist(),
            0.4,
            0.3
        )
        
        persons = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, x2, y2 = boxes[i]
                persons.append({
                    'bbox': (x, y, x2 - x, y2 - y),
                    'confidence': float(weights[i]),
                    'type': 'person'
                })
        
        return persons
    
    def detect_persons_mobilenet(self, image, confidence_threshold=0.5):
        if self.person_net is None:
            return self.detect_persons_hog(image)
        
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            0.007843,
            (300, 300),
            127.5
        )
        
        self.person_net.setInput(blob)
        detections = self.person_net.forward()
        
        persons = []
        for i in range(detections.shape[2]):
            class_id = int(detections[0, 0, i, 1])
            confidence = detections[0, 0, i, 2]
            
            if class_id == 15 and confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x2, y2) = box.astype("int")
                
                x = max(0, x)
                y = max(0, y)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                persons.append({
                    'bbox': (x, y, x2 - x, y2 - y),
                    'confidence': float(confidence),
                    'type': 'person'
                })
        
        return persons
    
    def merge_detections(self, faces, persons):
        merged = []
        used_persons = set()
        
        for face in faces:
            fx, fy, fw, fh = face['bbox']
            face_center_x = fx + fw // 2
            face_center_y = fy + fh // 2
            
            matched_person = None
            min_distance = float('inf')
            
            for i, person in enumerate(persons):
                if i in used_persons:
                    continue
                
                px, py, pw, ph = person['bbox']
                
                if (px <= face_center_x <= px + pw and 
                    py <= face_center_y <= py + ph // 2):
                    
                    distance = abs(face_center_y - py)
                    if distance < min_distance:
                        min_distance = distance
                        matched_person = i
            
            if matched_person is not None:
                used_persons.add(matched_person)
                merged.append({
                    'face': face['bbox'],
                    'body': persons[matched_person]['bbox'],
                    'confidence': (face['confidence'] + persons[matched_person]['confidence']) / 2,
                    'type': 'person_with_face'
                })
            else:
                merged.append({
                    'face': face['bbox'],
                    'body': None,
                    'confidence': face['confidence'],
                    'type': 'face_only'
                })
        
        for i, person in enumerate(persons):
            if i not in used_persons:
                merged.append({
                    'face': None,
                    'body': person['bbox'],
                    'confidence': person['confidence'],
                    'type': 'body_only'
                })
        
        return merged
    
    def draw_detections(self, image, merged_detections):
        for detection in merged_detections:
            if detection['type'] == 'person_with_face':
                bx, by, bw, bh = detection['body']
                cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (147, 20, 255), 3)
                
                fx, fy, fw, fh = detection['face']
                cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                
                label = f"Person {detection['confidence']:.2f}"
                cv2.putText(image, label, (bx, by - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (147, 20, 255), 2)
            
            elif detection['type'] == 'face_only':
                fx, fy, fw, fh = detection['face']
                cv2.rectangle(image, (fx, fy), (fx + fw, fy + fh), (0, 255, 0), 2)
                label = f"Face {detection['confidence']:.2f}"
                cv2.putText(image, label, (fx, fy - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            elif detection['type'] == 'body_only':
                bx, by, bw, bh = detection['body']
                cv2.rectangle(image, (bx, by), (bx + bw, by + bh), (255, 255, 0), 2)
                label = f"Person {detection['confidence']:.2f}"
                cv2.putText(image, label, (bx, by - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def process_image(self, image_path, output_path=None):
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        
        original = image.copy()
        
        print("\nDetecting faces...")
        faces = self.detect_faces(image)
        print(f"Found {len(faces)} face(s)")
        
        print("\nDetecting persons...")
        persons = self.detect_persons_mobilenet(image)
        print(f"Found {len(persons)} person(s)")
        
        print("\nMerging detections...")
        merged = self.merge_detections(faces, persons)
        print(f"Merged into {len(merged)} detection(s)")
        
        result_image = original.copy()
        self.draw_detections(result_image, merged)
        
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"\nOutput saved to: {output_path}")
        
        return result_image, original, merged
    
    def display_results(self, image, original):
        combined = np.hstack([original, image])
        
        scale = 1200 / combined.shape[1]
        if scale < 1:
            width = 1200
            height = int(combined.shape[0] * scale)
            combined = cv2.resize(combined, (width, height))
        
        cv2.imshow('Original vs Detected', combined)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Modern Human Detection using OpenCV DNN')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('-o', '--output', type=str, default=None, help='Path to save output')
    parser.add_argument('-d', '--display', action='store_true', help='Display results')
    
    args = parser.parse_args()
    
    detector = ModernHumanDetector()
    result = detector.process_image(args.image_path, args.output)
    
    if result:
        image, original, detections = result
        
        print("\n" + "=" * 50)
        print("DETECTION SUMMARY")
        print("=" * 50)
        for i, det in enumerate(detections):
            print(f"\nDetection {i + 1}:")
            print(f"  Type: {det['type']}")
            print(f"  Confidence: {det['confidence']:.2%}")
            if det['face']:
                print(f"  Face: {det['face']}")
            if det['body']:
                print(f"  Body: {det['body']}")
        print("=" * 50)
        
        if args.display:
            detector.display_results(image, original)

if __name__ == '__main__':
    main()