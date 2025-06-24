import os
import google_streetview.api
import requests
import math
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

# Define parameters
# best
latitude = 36.0159357
longitude = 129.3246659

# better
# latitude = 37.869260
# longitude = -122.254811

# good
# latitude = 37.5548238
# longitude = 126.9722165

# bad
# latitude = 46.414382
# longitude = 10.013988

# bad
# latitude = 40.720032
# # longitude = -73.988354

initial_heading = 165
pitch = 0
num_images = 20
distance_increment = 15  # Distance in meters for each step
api_key = 'AIzaSyDIyw5eharZPSRucW3_opA1lZg1FcwvdWU'  # Replace with your valid API key

# Create output directories if they don't exist
output_dir = 'output_gps'
seed_images_dir = os.path.join(output_dir, 'seed_images')
filtered_images_dir = os.path.join(output_dir, 'filtered_images')

for directory in [output_dir, seed_images_dir, filtered_images_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# List to store latitude and longitude points
lat_lon_points = []

def move_forward(lat, lon, heading, distance):
    """Calculate new lat/lon coordinates given a starting point, heading, and distance."""
    R = 6371000  # Earth's radius in meters

    # Convert heading to radians
    heading_rad = math.radians(heading)

    # Calculate the new latitude
    new_lat = lat + (distance / R) * (180 / math.pi) * math.cos(heading_rad)

    # Calculate the new longitude
    new_lon = lon + (distance / R) * (180 / math.pi) * math.sin(heading_rad) / math.cos(math.radians(lat))

    return new_lat, new_lon

def fetch_and_save_image(location, heading, index):
    params = [{
        'size': '640x640',  # Max 640x640 pixels
        'location': location,
        'heading': heading,
        'pitch': pitch,
        'key': api_key
    }]

    try:
        # Create a results object
        results = google_streetview.api.results(params)

        # Check the status in the metadata
        if results.metadata and results.metadata[0]['status'] == 'OK':
            print(f"Image {index+1} found for location {location}, proceeding to download.")

            # Download the image
            image_url = results.links[0]
            response = requests.get(image_url)
            if response.status_code == 200:
                filename = os.path.join(seed_images_dir, f'image_{index+1}.jpg')
                with open(filename, 'wb') as file:
                    file.write(response.content)
                print(f"Image {index+1} saved as {filename}")
                return True
            else:
                print(f"Failed to download image {index+1} from {image_url}")
        else:
            error_message = results.metadata[0].get('error_message', 'Unknown error')
            status = results.metadata[0].get('status', 'Unknown status')
            print(f"No image found for location {location}. Status: {status}, Error message: {error_message}")

    except Exception as e:
        print(f"Error fetching image {index+1} for location {location}: {e}")

    return False

# Fetch and save consecutive images by moving forward
lat, lon = latitude, longitude
for i in range(num_images):
    location = f"{lat},{lon}"
    if fetch_and_save_image(location, initial_heading, i):
        lat_lon_points.append((lat, lon))
    lat, lon = move_forward(lat, lon, initial_heading, distance_increment)

# Convert points to numpy array for further processing
lat_lon_array = np.array(lat_lon_points)

# Perform RANSAC regression
lats = lat_lon_array[:, 0].reshape(-1, 1)
lons = lat_lon_array[:, 1]
ransac = RANSACRegressor(base_estimator=LinearRegression(), min_samples=2, residual_threshold=1e-3, max_trials=1000)
ransac.fit(lats, lons)

# Best fit line coefficients
slope = ransac.estimator_.coef_[0]
intercept = ransac.estimator_.intercept_

# Calculate perpendicular distances from the best fit line
perpendicular_distances = np.abs(slope * lats.flatten() - lons + intercept) / np.sqrt(slope**2 + 1)

# Threshold distance to filter points
threshold = 0.0001

# Filter points based on the threshold
filtered_indices = [i for i, dist in enumerate(perpendicular_distances) if dist <= threshold]

# Print filtered points for verification
print("Filtered points:")
for i in filtered_indices:
    print(f"{lat_lon_points[i][0]}, {lat_lon_points[i][1]}")

# Save only the filtered images
for i in filtered_indices:
    src_filename = os.path.join(seed_images_dir, f'image_{i+1}.jpg')
    if os.path.exists(src_filename):
        dest_filename = os.path.join(filtered_images_dir, f'image_{i+1}.jpg')
        os.rename(src_filename, dest_filename)
        print(f"Image {i+1} saved as {dest_filename}")
