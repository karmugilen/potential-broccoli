import os
from PIL import Image

# Check what images are available in my_images directory
image_dir = "my_images"
image_extensions = ['.png', '.jpg', '.jpeg']

image_paths = []
for file in os.listdir(image_dir):
    for ext in image_extensions:
        if file.lower().endswith(ext.lower()):
            image_paths.append(os.path.join(image_dir, file))
            break  # Break to avoid adding same file multiple times

print(f"Found {len(image_paths)} images:")
for path in image_paths[:5]:  # Show first 5
    print(f"  {path}")
    try:
        img = Image.open(path)
        print(f"    Size: {img.size}, Mode: {img.mode}")
    except Exception as e:
        print(f"    Error loading: {e}")

print(f"\nTotal images found: {len(image_paths)}")
if len(image_paths) >= 2:
    print(f"Using {image_paths[0]} and {image_paths[1]} for testing")
else:
    print("Not enough images for testing")