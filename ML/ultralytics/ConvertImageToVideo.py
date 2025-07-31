import cv2
import os
from natsort import natsorted  # Optional, for natural sorting of file names

def images_to_video(image_folder, output_path, fps=30, video_format='mp4v'):
    # Get all image files from the folder
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Sort images (naturally: img1, img2, ..., img10)
    image_files = natsorted(image_files)

    if not image_files:
        print("No image files found in the folder.")
        return

    # Read the first image to get the frame size
    first_image_path = os.path.join(image_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*video_format)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Converting {len(image_files)} images to video...")

    for filename in image_files:
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Skipping file {filename} (couldn't read).")
            continue
        out.write(img)

    out.release()
    print(f"Video saved at {output_path}")

# Example usage
if __name__ == "__main__":
    image_folder = '/Users/adarshgowda/pro/PPE_5'  # e.g. './images'
    output_video = 'PPE5.mp4'
    fps = 1  # You can set your desired FPS here
    images_to_video(image_folder, output_video, fps)
