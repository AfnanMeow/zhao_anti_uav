import os
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# Paths
input_dir   = 'C:/Users/T25301039/Downloads/internet_drones'  # Change this to your folder path
output_dir  = 'C:/Users/T25301039/Downloads/internet_drones_results'

os.makedirs(output_dir, exist_ok=True)

print("Loading model...")
# Load model
uav_detection = pipeline(
    Tasks.domain_specific_object_detection,
    model='damo/cv_tinynas_uav-detection_damoyolo',
)
print("Model loaded successfully!")

# Get all image files
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_files = [f for f in os.listdir(input_dir) 
               if f.lower().endswith(image_extensions)]

print(f"Found {len(image_files)} images in {input_dir}")

saved_count = 0

for idx, img_file in enumerate(image_files):
    img_path = os.path.join(input_dir, img_file)
    
    # Read image
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Warning: Could not read {img_file}")
        continue
    
    # Resize to 640x512 if needed (optional - model can handle different sizes)
    # frame = cv2.resize(frame, (640, 512))
    
    # Run detection
    output = uav_detection(img_path)
    
    # Draw predicted boxes in RED
    if 'boxes' in output and len(output['boxes']) > 0:
        boxes  = output['boxes']
        scores = output['scores'] if 'scores' in output else [1.0] * len(boxes)
        
        for box, score in zip(boxes, scores):
            if float(score) < 0.3:  # Confidence threshold
                continue
            
            # Convert box coordinates to integers
            if len(box) == 4:
                x1, y1, x2, y2 = [int(v) for v in box]
                
                # Draw red rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add confidence score label
                label = f'Drone: {score:.2f}'
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Print detection info
        print(f"{img_file}: Found {len([s for s in scores if float(s) >= 0.3])} drone(s)")
    else:
        print(f"{img_file}: No drones detected")
    
    # Save output image
    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, frame)
    saved_count += 1
    
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1}/{len(image_files)} images...")

print(f'\nDone!')
print(f'Images processed: {saved_count}')
print(f'Output directory: {output_dir}')
print('Red boxes show detected drones with confidence scores')