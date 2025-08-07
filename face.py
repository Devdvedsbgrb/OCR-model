import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Нүүр ба нүд илрүүлэх Haar Cascade загваруудыг ачаалж байна.
class ImprovedFaceSimilarity:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # (reference image)-ийг ачаалж, саарал өнгөөр буулгаж бэлдэнэ.
    def load_reference_image(self, image_path):
        """Load and process reference image"""
        if not os.path.exists(image_path):
            print(f"Reference image not found: {image_path}")
            return None, None
        
        ref_img = cv2.imread(image_path)
        if ref_img is None:
            print(f"Could not load reference image: {image_path}")
            return None, None
        
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        return ref_img, ref_gray
    
    def detect_face_features(self, gray_image):
        """Detect face and extract features"""
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray_image, 1.1, 4)
        
        if len(faces) == 0:
            return None, None, None
        
        # Get the largest face (most prominent)
        largest_face = max(faces, key=lambda x: x[2] * x[3])
        (x, y, w, h) = largest_face
        
        # Extract face region
        face_roi = gray_image[y:y+h, x:x+w]
        
        # Detect eyes within the face
        eyes = self.eye_cascade.detectMultiScale(face_roi)
        
        return largest_face, face_roi, eyes
    
    def calculate_face_similarity(self, face1, face2):
        """Calculate face similarity using multiple methods"""
        if face1 is None or face2 is None:
            return 0.0
        
        # Resize both faces to same size for comparison
        face1_resized = cv2.resize(face1, (100, 100))
        face2_resized = cv2.resize(face2, (100, 100))
        
        # Method 1: Histogram comparison
        hist1 = cv2.calcHist([face1_resized], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([face2_resized], [0], None, [256], [0, 256])
        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        # Method 2: Structural Similarity Index (SSIM)
        # Convert to float for better precision
        face1_float = face1_resized.astype(np.float32) / 255.0
        face2_float = face2_resized.astype(np.float32) / 255.0
        
        # Calculate SSIM manually
        mu1 = np.mean(face1_float)
        mu2 = np.mean(face2_float)
        
        sigma1_sq = np.var(face1_float)
        sigma2_sq = np.var(face2_float)
        sigma12 = np.mean((face1_float - mu1) * (face2_float - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        # Method 3: Normalized Cross Correlation
        face1_norm = (face1_float - np.mean(face1_float)) / np.std(face1_float)
        face2_norm = (face2_float - np.mean(face2_float)) / np.std(face2_float)
        
        ncc = np.mean(face1_norm * face2_norm)
        
        # Combine all methods
        hist_weight = 0.3
        ssim_weight = 0.5
        ncc_weight = 0.2
        
        # Normalize and combine scores
        hist_score = max(0, hist_similarity) * 100
        ssim_score = max(0, ssim) * 100
        ncc_score = max(0, ncc) * 100
        
        final_similarity = (hist_score * hist_weight + 
                          ssim_score * ssim_weight + 
                          ncc_score * ncc_weight)
        
        return final_similarity
    
    def run_face_comparison(self, reference_image_path):
        # Load reference image
        ref_img, ref_gray = self.load_reference_image(reference_image_path)
        if ref_img is None:
            return
        
        # Detect face in reference image
        ref_face_info = self.detect_face_features(ref_gray)
        if ref_face_info[0] is None:
            print("Жишээ зурагт нүүр олдсонгүй!")
            return
        
        ref_face_bbox, ref_face_roi, ref_eyes = ref_face_info
        print(f"Жишээ зурагт нүүр олдлоо: {ref_face_bbox}")
        
        # Start camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Камер нээх боломжгүй байна!")
            return
        
        print("\nУдирдлага:")
        print("'a' - Нүүр харьцуулах")
        print("'q' - Гарах")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Камераас зураг авч чадсангүй!")
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect face in current frame
            current_face_info = self.detect_face_features(gray)
            
            # Display instructions
            cv2.putText(frame, "Press 'a' to compare faces, 'q' to quit", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw rectangle around detected face
            if current_face_info[0] is not None:
                (x, y, w, h) = current_face_info[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Face detected", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Face Comparison - Press a to compare, q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('a'):
                if current_face_info[0] is None:
                    print("Одоогийн зурагт нүүр олдсонгүй!")
                    continue
                
                # Calculate similarity
                similarity = self.calculate_face_similarity(ref_face_roi, current_face_info[1])
                
                # Show detailed analysis
                self.show_similarity_analysis(similarity)
                
                # Draw result on frame
                (x, y, w, h) = current_face_info[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Add similarity text
                color = (0, 255, 0) if similarity > 50 else (0, 0, 255)
                cv2.putText(frame, f"Similarity: {similarity:.1f}%", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Show result
                cv2.imshow('Face Comparison Result', frame)
                cv2.waitKey(3000)  # Show for 3 seconds
                
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_similarity_analysis(self, similarity):
        
        print(f"Харьцуулалтын үр дүн: {similarity:.2f}%")
        
        # Interpret similarity
        if similarity >= 80:
            print(" Маш их төстэй!")
        elif similarity >= 60:
            print(" Төстэй.")
        elif similarity >= 40:
            print("Ижил хүн байж болох юм.")
        elif similarity >= 20:
            print("Өөр хүн")
        else:
            print("Төстэй биш.")

def main():
    """Main function"""
    app = ImprovedFaceSimilarity()
    
    # Use reference image
    reference_image = "images/passport.jpg"  # You can change this to your reference image
    
    print("Нүүр харьцуулах програм эхэллээ...")
    print(f"Жишээ зураг: {reference_image}")
    
    app.run_face_comparison(reference_image)

if __name__ == "__main__":
    main()