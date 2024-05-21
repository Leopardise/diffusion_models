import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_and_match_features(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return kp1, kp2, matches

def estimate_motion_vanishing_angle(kp1, kp2, matches):
    if len(matches) < 8:
        raise ValueError("Not enough matches to compute motion")
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2)
    return R, t

def reconstruct_path(image_dir):
    path = [np.array([0, 0, 0])]  # Starting at origin
    current_pos = np.array([0, 0, 0])
    current_orientation = np.eye(3)  # Initial orientation is the identity matrix

    for sequence_id in os.listdir(image_dir):
        sequence_path = os.path.join(image_dir, sequence_id)
        if not os.path.isdir(sequence_path):
            continue  # Skip non-directory files

        image_files = sorted(os.listdir(sequence_path))

        for i in range(len(image_files) - 1):
            img1_path = os.path.join(sequence_path, image_files[i])
            img2_path = os.path.join(sequence_path, image_files[i + 1])

            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

            kp1, kp2, matches = detect_and_match_features(img1, img2)

            try:
                R, t = estimate_motion_vanishing_angle(kp1, kp2, matches)
                # Update the current orientation
                current_orientation = current_orientation @ R
                # Calculate the new position
                current_pos = current_pos + current_orientation @ t.reshape(-1)
                # Append the new position to the path
                path.append(current_pos)
            except ValueError as e:
                print(f"Error processing images {image_files[i]} and {image_files[i + 1]}: {e}")

    return np.array(path)

def plot_path(path):
    plt.plot(path[:, 0], path[:, 2])
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.title('Reconstructed Camera Path')
    plt.show()

# Main execution
image_dir = 'output'
path = reconstruct_path(image_dir)
plot_path(path)
