import argparse
import cv2
import numpy as np
from itertools import combinations
import os
from glob import glob

class VPDetection(object):
    def __init__(self, length_thresh=30, principal_point=None, focal_length=1500, seed=None):
        self._length_thresh = length_thresh
        self._principal_point = principal_point
        self._focal_length = focal_length
        self._angle_thresh = np.pi / 30  # For displaying debug image
        self._vps = None  # For storing the VPs in 3D space
        self._vps_2D = None  # For storing the VPs in 2D space
        self.__img = None  # Stores the image locally
        self.__clusters = None  # Stores which line index maps to what VP
        self.__tol = 1e-8  # Tolerance for floating point comparison
        self.__angle_tol = np.pi / 3  # (pi / 180 * (60 degrees)) = +/- 30 deg
        self.__lines = None  # Stores the line detections internally
        self.__zero_value = 0.001  # Threshold to check augmented coordinate
        self.__seed = seed  # Set seed for reproducibility
        noise_ratio = 0.5  # Outlier/inlier ratio for RANSAC estimation
        p = (1.0 / 3.0) * ((1.0 - noise_ratio) ** 2.0)
        conf = 0.9999
        self.__ransac_iter = int(np.log(1 - conf) / np.log(1.0 - p))

    @property
    def length_thresh(self):
        return self._length_thresh

    @length_thresh.setter
    def length_thresh(self, value):
        if value <= 0:
            raise ValueError('Invalid threshold: {}'.format(value))
        self._length_thresh = value

    @property
    def principal_point(self):
        return self._principal_point

    @principal_point.setter
    def principal_point(self, value):
        try:
            assert isinstance(value, (list, tuple)) and not isinstance(value, str)
            assert len(value) == 2
        except AssertionError:
            raise ValueError('Invalid principal point: {}'.format(value))
        self._length_thresh = value

    @property
    def focal_length(self):
        return self._focal_length

    @focal_length.setter
    def focal_length(self, value):
        if value < self.__tol:
            raise ValueError('Invalid focal length: {}'.format(value))
        self._focal_length = value

    @property
    def vps(self):
        return self._vps

    @property
    def vps_2D(self):
        return self._vps_2D

    def __detect_lines(self, img):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img

        # Apply Gaussian Blur
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 1.4)

        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=self._length_thresh, maxLineGap=10)

        if lines is not None:
            lines = lines[:, 0, :]  # Reshape from (N, 1, 4) to (N, 4)
            self.__lines = lines
        else:
            self.__lines = np.array([])

        return self.__lines

    def __find_vp_hypotheses_two_lines(self):
        N = self.__lines.shape[0]
        if N < 2:
            print("Not enough lines detected for vanishing point hypothesis generation.")
            return np.array([])  # Return an empty array if not enough lines

        num_bins_vp2 = 360
        vp2_step = np.pi / 180.0
        p1 = np.column_stack((self.__lines[:, :2], np.ones(N, dtype=np.float32)))
        p2 = np.column_stack((self.__lines[:, 2:], np.ones(N, dtype=np.float32)))
        cross_p = np.cross(p1, p2)
        dx = p1[:, 0] - p2[:, 0]
        dy = p1[:, 1] - p2[:, 1]
        lengths = np.sqrt(dx * dx + dy * dy)
        orientations = np.arctan2(dy, dx)
        orientations[orientations < 0] = orientations[orientations < 0] + np.pi
        self.__cross_p = cross_p
        self.__lengths = lengths
        self.__orientations = orientations
        vp_hypos = np.zeros((self.__ransac_iter * num_bins_vp2, 3, 3), dtype=np.float32)
        i = 0
        if self.__seed is not None:
            gen = np.random.RandomState(self.__seed)
        while i < self.__ransac_iter:
            if self.__seed is not None:
                (idx1, idx2) = gen.permutation(N)[:2]
            else:
                (idx1, idx2) = np.random.permutation(N)[:2]
            vp1_img = np.cross(cross_p[idx1], cross_p[idx2])
            if np.abs(vp1_img[2]) < self.__tol:
                continue
            vp1 = np.zeros(3, dtype=np.float32)
            vp1[:2] = vp1_img[:2] / vp1_img[2] - self._principal_point
            vp1[2] = self._focal_length
            vp1 /= np.sqrt(np.sum(np.square(vp1)))
            la = np.arange(num_bins_vp2) * vp2_step
            kk = vp1[0] * np.sin(la) + vp1[1] * np.cos(la)
            phi = np.arctan(-vp1[2] / kk)
            vp2 = np.column_stack([
                np.sin(phi) * np.sin(la),
                np.sin(phi) * np.cos(la),
                np.cos(phi)
            ])
            vp2[np.abs(vp2[:, 2]) < self.__tol, 2] = self.__zero_value
            vp2 /= np.sqrt(np.sum(np.square(vp2), axis=1, keepdims=True))
            vp2[vp2[:, 2] < 0, :] *= -1.0
            vp3 = np.cross(vp1, vp2)
            vp3[np.abs(vp3[:, 2]) < self.__tol, 2] = self.__zero_value
            vp3 /= np.sqrt(np.sum(np.square(vp3), axis=1, keepdims=True))
            vp3[vp3[:, 2] < 0, :] *= -1.0
            vp_hypos[i * num_bins_vp2:(i + 1) * num_bins_vp2, 0, :] = vp1
            vp_hypos[i * num_bins_vp2:(i + 1) * num_bins_vp2, 1, :] = vp2
            vp_hypos[i * num_bins_vp2:(i + 1) * num_bins_vp2, 2, :] = vp3
            i += 1
        return vp_hypos

    def __get_sphere_grids(self):
        bin_size = np.pi / 180.0
        lat_span = np.pi / 2.0
        long_span = 2.0 * np.pi
        num_bins_lat = int(lat_span / bin_size)
        num_bins_lon = int(long_span / bin_size)
        combos = list(combinations(range(self.__lines.shape[0]), 2))
        combos = np.asarray(combos, dtype=int)
        pt_intersect = np.cross(self.__cross_p[combos[:, 0]], self.__cross_p[combos[:, 1]])
        mask = np.abs(pt_intersect[:, 2]) >= self.__tol
        ang = np.abs(self.__orientations[combos[:, 0]] - self.__orientations[combos[:, 1]])
        ang = np.minimum(np.pi - ang, ang)
        mask = np.logical_and(mask, np.abs(ang) <= self.__angle_tol)
        pt_intersect = pt_intersect[mask]
        ang = ang[mask]
        combos = combos[mask]
        X = (pt_intersect[:, 0] / pt_intersect[:, 2]) - self._principal_point[0]
        Y = (pt_intersect[:, 1] / pt_intersect[:, 2]) - self._principal_point[1]
        Z = self._focal_length
        lat = np.arccos(Z / np.sqrt(X * X + Y * Y + Z * Z))
        lon = np.arctan2(X, Y) + np.pi
        la_bin = (lat / bin_size).astype(int)
        lon_bin = (lon / bin_size).astype(int)
        la_bin[la_bin >= num_bins_lat] = num_bins_lat - 1
        lon_bin[lon_bin >= num_bins_lon] = num_bins_lon - 1
        bin_num = la_bin * num_bins_lon + lon_bin
        weights = np.sqrt(
            self.__lengths[combos[:, 0]] * self.__lengths[combos[:, 1]]) * (np.sin(2.0 * ang) + 0.2)
        sphere_grid = np.bincount(
            bin_num, weights=weights, minlength=num_bins_lat * num_bins_lon).reshape(
            (num_bins_lat, num_bins_lon)).astype(np.float32)
        sphere_grid += cv2.filter2D(sphere_grid, -1, (1.0 / 9.0) * np.ones(
            (3, 3)))
        return sphere_grid

    def __get_best_vps_hypo(self, sphere_grid, vp_hypos):
        if vp_hypos.size == 0:
            print("No vanishing point hypotheses were generated.")
            return np.array([])

        N = vp_hypos.shape[0]
        bin_size = np.pi / 180.0
        mask = np.logical_and(
            np.abs(vp_hypos[:, :, 2]) >= self.__tol,
            np.abs(vp_hypos[:, :, 2]) <= 1.0)
        ids = np.arange(N).astype(int)
        ids = np.column_stack([ids, ids, ids])
        ids = ids[mask]
        lat = np.arccos(vp_hypos[:, :, 2][mask])
        lon = np.arctan2(vp_hypos[:, :, 0][mask], vp_hypos[:, :, 1][mask]) + np.pi
        la_bin = (lat / bin_size).astype(int)
        lon_bin = (lon / bin_size).astype(int)
        la_bin[la_bin == 90] = 89
        lon_bin[lon_bin == 360] = 359
        weights = sphere_grid[la_bin, lon_bin]
        votes = np.bincount(ids, weights=weights, minlength=N).astype(np.float32)
        best_idx = np.argmax(votes)
        final_vps = vp_hypos[best_idx]
        vps_2D = self._focal_length * (final_vps[:, :2] / final_vps[:, 2][:, None])
        vps_2D += self._principal_point
        z_idx = np.argmax(np.abs(vps_2D[:, 1]))
        ind = np.arange(3).astype(int)
        mask = np.ones(3, dtype=bool)
        mask[z_idx] = False
        ind = ind[mask]
        vps_trim = vps_2D[mask]
        x_idx = np.argmin(vps_trim[:, 0])
        x_idx = ind[x_idx]
        mask[x_idx] = False
        x2_idx = np.argmax(mask)
        final_vps = final_vps[[x2_idx, x_idx, z_idx], :]
        vps_2D = vps_2D[[x2_idx, x_idx, z_idx], :]
        self._vps = final_vps
        self._vps_2D = vps_2D
        return final_vps

    def __cluster_lines(self, vps_hypos):
        x1 = self.__lines[:, 0].astype(np.float64)
        y1 = self.__lines[:, 1].astype(np.float64)
        x2 = self.__lines[:, 2].astype(np.float64)
        y2 = self.__lines[:, 3].astype(np.float64)
        xc = (x1 + x2) / 2.0
        yc = (y1 + y2) / 2.0
        dx = x1 - x2
        dy = y1 - y2
        norm_factor = np.sqrt(dx * dx + dy * dy)
        dx /= norm_factor
        dy /= norm_factor
        xp = self._vps_2D[:, 0][:, None] - xc[None]
        yp = self._vps_2D[:, 1][:, None] - yc[None]
        norm_factor = np.sqrt(xp * xp + yp * yp)
        xp /= norm_factor
        yp /= norm_factor
        dotp = dx[None] * xp + dy[None] * yp
        dotp[dotp > 1.0] = 1.0
        dotp[dotp < -1.0] = -1.0
        ang = np.arccos(dotp)
        ang = np.minimum(np.pi - ang, ang)
        min_ang = np.min(ang, axis=0)
        idx_ang = np.argmin(ang, axis=0)
        mask = min_ang <= self._angle_thresh
        self.__clusters = [
            np.where(np.logical_and(mask, idx_ang == i))[0] for i in range(3)
        ]

    def find_vps(self, img):
        if isinstance(img, str):
            img = cv2.imread(img, -1)
            if img is None:
                raise FileNotFoundError(f"Cannot open/read file: {img}")
        self.__img = img  # Keep a copy for later
        if self._principal_point is None:
            rows, cols = img.shape[:2]
            self._principal_point = np.array([cols / 2.0, rows / 2.0], dtype=np.float32)
        _ = self.__detect_lines(img)
        if self.__lines.size == 0:
            print(f"No lines detected in image: {img}")
            return np.array([])

        vps_hypos = self.__find_vp_hypotheses_two_lines()
        if vps_hypos.size == 0:
            print(f"No VP hypotheses generated for image: {img}")
            return np.array([])

        self.__vps_hypos = vps_hypos  # Save a copy
        sphere_grid = self.__get_sphere_grids()
        best_vps = self.__get_best_vps_hypo(sphere_grid, vps_hypos)
        if best_vps.size == 0:
            print(f"No best VP hypotheses found for image: {img}")
            return np.array([])

        self.__final_vps = best_vps  # Save a copy
        self.__clusters = None  # Reset because of new image
        return best_vps

    def create_debug_VP_image(self, show_image=False, save_image=None):
        if self.__clusters is None:
            self.__cluster_lines(self.__vps_hypos)
        if save_image is not None and not isinstance(save_image, str):
            raise ValueError('The save_image path should be a string')
        img = self.__img.copy()
        if len(img.shape) == 2:  # If grayscale, artificially make into RGB
            img = np.dstack([img, img, img])
        colours = 255 * np.eye(3)
        colours = colours[:, ::-1].astype(int).tolist()
        all_clusters = np.hstack(self.__clusters)
        status = np.ones(self.__lines.shape[0], dtype=bool)
        status[all_clusters] = False
        ind = np.where(status)[0]
        for (x1, y1, x2, y2) in self.__lines[ind]:
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 0), 2, cv2.LINE_AA)
        for i in range(3):
            for (x1, y1, x2, y2) in self.__lines[self.__clusters[i]]:
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), colours[i], 2, cv2.LINE_AA)
        if show_image:
            cv2.imshow('VP Debug Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if save_image is not None and save_image != '':
            cv2.imwrite(save_image, img)
        return img

def process_folder(folder_path, length_thresh, principal_point, focal_length, debug, debug_show, debug_path, seed):
    image_paths = sorted(glob(os.path.join(folder_path, '*.jpg'))) + sorted(glob(os.path.join(folder_path, '*.png')))
    if len(image_paths) == 0:
        raise ValueError("The folder must contain at least one image.")

    vpd = VPDetection(length_thresh=length_thresh, principal_point=principal_point, focal_length=focal_length, seed=seed)
    for image_path in image_paths:
        # Detect vanishing points
        vps = vpd.find_vps(image_path)

        if vps.size > 0:
            print(f"\nVanishing Points for Image {os.path.basename(image_path)} in 3D space:")
            for i, vp in enumerate(vps):
                print(f"Vanishing Point {i + 1}: {vp}")

            print(f"\nVanishing Points for Image {os.path.basename(image_path)} in image coordinates:")
            for i, vp in enumerate(vpd.vps_2D):
                print(f"Vanishing Point {i + 1}: {vp}")
        else:
            print(f"No vanishing points detected for image: {os.path.basename(image_path)}")

        if debug or debug_show:
            vpd.create_debug_VP_image(debug_show, os.path.join(debug_path, f"debug_{os.path.basename(image_path)}"))

def main():
    parser = argparse.ArgumentParser(description="Vanishing Point Detection for a Folder of Images")
    parser.add_argument('-f', '--folder-path', required=True, help='Path to the folder containing the images')
    parser.add_argument('-lt', '--length-thresh', default=30, type=float, help='Minimum line length (in pixels) for detecting lines')
    parser.add_argument('-pp', '--principal-point', default=None, nargs=2, type=float, help='Principal point of the camera (default is image centre)')
    parser.add_argument('-fl', '--focal-length', default=1500, type=float, help='Focal length of the camera (in pixels)')
    parser.add_argument('-d', '--debug', action='store_true', help='Turn on debug image mode')
    parser.add_argument('-ds', '--debug-show', action='store_true', help='Show the debug image in an OpenCV window')
    parser.add_argument('-dp', '--debug-path', default="debug_images", help='Path for writing the debug images')
    parser.add_argument('-s', '--seed', default=None, type=int, help='Specify random seed for reproducible results')
    args = parser.parse_args()

    # Ensure debug image path exists
    if args.debug or args.debug_show:
        os.makedirs(args.debug_path, exist_ok=True)

    try:
        process_folder(args.folder_path, args.length_thresh, args.principal_point, args.focal_length, args.debug, args.debug_show, args.debug_path, args.seed)
    except FileNotFoundError as e:
        print(e)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
