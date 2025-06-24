import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random
import math

# Helper functions

# Show images given a list of images
def show_images(image):
    plt.figure()
    plt.imshow(image, cmap='gray')

# Load images from a folder given their filenames
def load_images(filepath):
    try:
        img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
        return img
    except IOError:
        print("File is not an image\n")
        exit()

# Plot lines on original images
def show_lines(image, lines, output_path):
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(image, pt1, pt2, (255, 0, 0), 1)  # Thinner lines with thickness 1
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Plot lines and points on original images
def show_point(image, point, output_path):
    cv2.circle(image, point, 3, (0, 255, 0), thickness=3)
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# Non-maximum suppression
def non_max_suppression(gradient_magnitude, gradient_angle):
    M, N = gradient_magnitude.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = gradient_angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = gradient_magnitude[i, j+1]
                    r = gradient_magnitude[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = gradient_magnitude[i+1, j-1]
                    r = gradient_magnitude[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = gradient_magnitude[i+1, j]
                    r = gradient_magnitude[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = gradient_magnitude[i-1, j-1]
                    r = gradient_magnitude[i+1, j+1]

                if (gradient_magnitude[i,j] >= q) and (gradient_magnitude[i,j] >= r):
                    Z[i,j] = gradient_magnitude[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass

    return Z

# Double threshold
def threshold(image, low, high):
    high_threshold = image.max() * high
    low_threshold = high_threshold * low

    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(image >= high_threshold)
    zeros_i, zeros_j = np.where(image < low_threshold)

    weak_i, weak_j = np.where((image <= high_threshold) & (image >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

# Edge tracking by hysteresis
def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i,j] == weak):
                try:
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

# Detect lines in the image
def detect_lines(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gau_kernel = cv2.getGaussianKernel(110, 4)
    gau_kern2d = np.outer(gau_kernel, gau_kernel)
    gau_kern2d = gau_kern2d / gau_kern2d.sum()
    blur_image = cv2.filter2D(gray_image, -1, gau_kern2d)

    # Compute gradients
    gradient_x = cv2.Sobel(blur_image, cv2.CV_64F, 1, 0, ksize=5)
    gradient_y = cv2.Sobel(blur_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_angle = np.arctan2(gradient_y, gradient_x)

    # Non-maximum suppression
    non_max_img = non_max_suppression(gradient_magnitude, gradient_angle)

    # Double threshold and Hysteresis
    threshold_img, weak, strong = threshold(non_max_img, 0.05, 0.15)
    edge_image = hysteresis(threshold_img, weak, strong)

    lines = cv2.HoughLines(edge_image.astype(np.uint8), 1, np.pi/120, 55)
    valid_lines = []
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            if (theta>0.08 and theta < 1.49) or (theta > 1.65 and theta < 3.06):
                valid_lines.append(line)
    return blur_image, edge_image, valid_lines

# Find the intersection point
def find_intersection_point(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    det_A = np.linalg.det(A)
    if det_A != 0:
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        return x0, y0
    else:
        return None

# Find the distance from a point to a line
def find_dist_to_line(point, line):
    x0, y0 = point
    rho, theta = line[0]
    m = (-1 * (np.cos(theta))) / np.sin(theta)
    c = rho / np.sin(theta)
    x = (x0 + m * y0 - m * c) / (1 + m**2)
    y = (m * x0 + (m**2) * y0 - (m**2) * c) / (1 + m**2) + c
    dist = math.sqrt((x - x0)**2 + (y - y0)**2)
    return dist

# RANSAC loop
def RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio):
    inlier_count_ratio = 0.
    vanishing_point = (0, 0)
    for iteration in range(ransac_iterations):
        selected_lines = random.sample(lines, 2)
        line1 = selected_lines[0]
        line2 = selected_lines[1]
        intersection_point = find_intersection_point(line1, line2)
        if intersection_point is not None:
            inlier_count = 0
            for line in lines:
                dist = find_dist_to_line(intersection_point, line)
                if dist < ransac_threshold:
                    inlier_count += 1
            if inlier_count / float(len(lines)) > inlier_count_ratio:
                inlier_count_ratio = inlier_count / float(len(lines))
                vanishing_point = intersection_point
            if inlier_count > len(lines) * ransac_ratio:
                break
    return vanishing_point

# Main function
ransac_iterations, ransac_threshold, ransac_ratio = 800, 13, 0.93

def process_images(input_folder, output_lines_folder, output_vp_folder, edge_output_folder):
    os.makedirs(output_lines_folder, exist_ok=True)
    os.makedirs(output_vp_folder, exist_ok=True)
    os.makedirs(edge_output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            image = load_images(image_path)
            blur_image, edge_image, lines = detect_lines(image)

            output_lines_path = os.path.join(output_lines_folder, filename)
            output_vp_path = os.path.join(output_vp_folder, filename)
            edge_output_path = os.path.join(edge_output_folder, filename)

            if len(lines) >= 2:  # Ensure there are enough lines for RANSAC
                vanishing_point = RANSAC(lines, ransac_iterations, ransac_threshold, ransac_ratio)
                show_point(image.copy(), vanishing_point, output_vp_path)
                show_lines(image, lines, output_lines_path)
            else:
                print(f"Not enough valid lines found in image: {filename}")
            cv2.imwrite(edge_output_path, edge_image)

if __name__ == "__main__":
    input_folder = 'output_gps/filtered_images'
    output_lines_folder = 'output_gps/detected_lines'
    output_vp_folder = 'output_gps/estimated_vanishing_point'
    edge_output_folder = 'output_gps/edge_detection'

    process_images(input_folder, output_lines_folder, output_vp_folder, edge_output_folder)
