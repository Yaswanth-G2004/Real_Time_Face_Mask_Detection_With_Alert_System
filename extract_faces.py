import os
import cv2
import xml.etree.ElementTree as ET

# Create output folders
os.makedirs('dataset/with_mask', exist_ok=True)
os.makedirs('dataset/without_mask', exist_ok=True)

# Path to annotation and image folder
annotation_dir = 'annotations'
image_dir = 'images'

count = 0

for file in os.listdir(annotation_dir):
    if not file.endswith('.xml'):
        continue

    xml_path = os.path.join(annotation_dir, file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find('filename').text
    img_path = os.path.join(image_dir, filename)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Image not found: {img_path}")
        continue

    for obj in root.findall('object'):
        label = obj.find('name').text
        if label not in ['with_mask', 'without_mask']:
            continue  # skip 'mask_weared_incorrect'

        bbox = obj.find('bndbox')
        x1 = int(float(bbox.find('xmin').text))
        y1 = int(float(bbox.find('ymin').text))
        x2 = int(float(bbox.find('xmax').text))
        y2 = int(float(bbox.find('ymax').text))

        # Crop and resize face
        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (128, 128))

        save_dir = f"dataset/{label}"
        save_path = os.path.join(save_dir, f"{label}_{count}.jpg")
        cv2.imwrite(save_path, face)
        count += 1

print("✅ All valid faces extracted and saved to dataset/")
