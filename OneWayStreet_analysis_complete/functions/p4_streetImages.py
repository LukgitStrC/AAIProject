import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class DepthEstimationModel:
  """
  Class for access to the depth anything model
  use predict to generate a depth map of an image
  """
  def __init__(self):
      if torch.backends.mps.is_available():
          self.device = torch.device("mps")
          print("Using Apple Silicon MPS for Depth Estimation.")
      else:
          self.device = torch.device("cpu")
      # Load the lightweight Depth Anything model
      model_name = "LiheYoung/depth-anything-small-hf"
      self.processor = AutoImageProcessor.from_pretrained(model_name)
      self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
      self.model.eval()

  def predict(self, image_input):
      """Generates a depth map from an image."""
      if isinstance(image_input, str):
          image = Image.open(image_input).convert("RGB")
      else:
          image = Image.fromarray(image_input[..., ::-1]) # Convert OpenCV BGR to RGB

      inputs = self.processor(images=image, return_tensors="pt")
      inputs = {k: v.to(self.device) for k, v in inputs.items()}

      with torch.no_grad():
          outputs = self.model(**inputs)
          predicted_depth = outputs.predicted_depth

      # Resize the output depth map to match the original image dimensions
      depth_map = torch.nn.functional.interpolate(
          predicted_depth.unsqueeze(1),
          size=image.size[::-1],
          mode="bicubic",
          align_corners=False,
      ).squeeze().cpu().numpy()

      return depth_map
    
def extract_perspective(equi_image, yaw, pitch, fov, width=600, height=600):
  """
  Extracts a planar perspective image from an equirectangular 360 panorama.
  
  Parameters:
  - equi_image: The original 360 image (numpy array from cv2.imread).
  - yaw: The horizontal viewing angle in degrees (-180 to 180).
  - pitch: The vertical viewing angle in degrees (0 is straight ahead).
  - fov: The horizontal Field of View in degrees (e.g., 90).
  - width, height: The dimensions of the output image.
  """
  # 1. Convert angles from degrees to radians
  yaw_rad = np.radians(yaw)
  pitch_rad = np.radians(pitch)
  fov_rad = np.radians(fov)
  
  equi_h, equi_w = equi_image.shape[:2]
  
  # 2. Calculate the focal length
  f = (width / 2.0) / np.tan(fov_rad / 2.0)
  
  # 3. Create a grid of x, y coordinates for the output image
  x, y = np.meshgrid(np.arange(width), np.arange(height))
  
  # Center the coordinates around 0,0
  x = x - width / 2.0
  y = y - height / 2.0
  z = np.full_like(x, f) # Depth is the focal length
  
  # 4. Create 3D rays (stack x, y, z into a single array)
  rays = np.stack((x, y, z), axis=-1)
  
  # 5. Build rotation matrices for pitch (X-axis) and yaw (Y-axis)
  Rx = np.array([
      [1, 0, 0],
      [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
      [0, np.sin(pitch_rad), np.cos(pitch_rad)]
  ])
  
  Ry = np.array([
      [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
      [0, 1, 0],
      [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
  ])
  
  # Combine rotations (Yaw then Pitch)
  R = Ry @ Rx
  
  # 6. Apply rotation to our 3D rays
  # Flatten the rays for matrix multiplication, then reshape back
  rays_flat = rays.reshape(-1, 3)
  rotated_rays_flat = rays_flat @ R.T
  rotated_rays = rotated_rays_flat.reshape(height, width, 3)
  
  rx, ry, rz = rotated_rays[:, :, 0], rotated_rays[:, :, 1], rotated_rays[:, :, 2]
  
  # 7. Convert the rotated 3D rays into spherical coordinates (Longitude, Latitude)
  theta = np.arctan2(rx, rz)
  phi = np.arcsin(ry / np.linalg.norm(rotated_rays, axis=-1))
  
  # 8. Map the spherical coordinates to the equirectangular image pixels
  # Map theta [-pi, pi] to [0, equi_w - 1]
  map_x = (theta / (2 * np.pi) + 0.5) * (equi_w - 1)
  # Map phi [-pi/2, pi/2] to [0, equi_h - 1]
  map_y = (phi / np.pi + 0.5) * (equi_h - 1)
  
  # 9. Sample the pixels using OpenCV's remap function
  # BORDER_WRAP allows the image to seamlessly wrap around the 360-degree edges
  planar_image = cv2.remap(
      equi_image, 
      map_x.astype(np.float32), 
      map_y.astype(np.float32), 
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_WRAP 
  )
  return planar_image

def find_circular_peaks(profile, distance, prominence):
  """
  Finds peaks in a circular 1D array.
  Parameters:
  - profile: array of values in each direction
  - distance: minimal distance between peaks (absolute)
  - prominence: to adjust how large the peak must be
  Returns: array of peak indices
  """
  W = len(profile)
  extended_profile = np.tile(profile, 3)
  extended_peaks, _ = find_peaks(
      extended_profile, 
      distance=distance, 
      prominence=prominence
  )
  valid_peaks = []
  for p in extended_peaks:
      if W <= p < 2 * W:
          valid_peaks.append(p % W)
  return np.array(valid_peaks)

def find_street_angles(depth_map):
  """
  Analyzes the depth mapto find the yaw angles of street entrances.
  Wraps find_circular_peaks by applying smoothing to the curve
  """
  H, W = depth_map.shape
  
  # only the middle of the image is analyzed to remove curvatures a the bottom and top
  # assumes that the image was taken horizontally
  top_crop = int(H * 0.35)
  bottom_crop = int(H * 0.65)
  horizon_band = depth_map[top_crop:bottom_crop, :]
  
  # Average and invert
  column_depths = np.mean(horizon_band, axis=0)
  max_depth_val = np.max(column_depths)
  inverted_profile = max_depth_val - column_depths
  
  # Smooth and find peaks
  depth_profile = gaussian_filter1d(inverted_profile, sigma=W/150)
  depth_peaks= find_circular_peaks(depth_profile, distance=W//12, prominence=np.max(depth_profile) * 0.1)
  
  # converts the coordinates back to angles
  yaws = [(x / W) * 360.0 - 180.0 for x in depth_peaks]
  
  return yaws

def extract_street_views(panorama_image, angles):
  """
  Extracts the images from the panorama according to the given angles
  Returns: list of extracted planar images
  """
  extracted_images = []
  print(f"Found {len(angles)} street entrances at angles: {angles}")
  for angle in angles:
      planar_image = extract_perspective(panorama_image, yaw=angle, pitch=0, fov=90, width=600, height=600)
      extracted_images.append(planar_image)
  return extracted_images

def display_extracted_streets(images, yaws=None):
  """
  Displays a list of extracted street view images using matplotlib.
  Automatically converts OpenCV BGR format to Matplotlib RGB format.
  
  Parameters:
  - images: List of numpy arrays (the extracted 600x600 images).
  - yaws: Optional list of angles corresponding to the images for titles.
  """
  num_images = len(images)
  
  if num_images == 0:
    print("No images to display.")
    return

  fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

  if num_images == 1:
    axes = [axes]

  for i, ax in enumerate(axes):
      # Convert BGR (OpenCV) to RGB (Matplotlib)
      img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
      
      # Display the image
      ax.imshow(img_rgb)
      ax.axis('off')
      if yaws and i < len(yaws):
        ax.set_title(f"Street View {i+1}\n(Angle: {yaws[i]:.1f}°)")
      else:
        ax.set_title(f"Street View {i+1}")
  plt.tight_layout()
  plt.show()

