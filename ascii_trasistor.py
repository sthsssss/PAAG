import numpy as np
import cv2
import math
from ultralytics import YOLO

class ASCIIArtGenerator:
    def __init__(self):
        # UTF-8 diagonal/slash characters for different angles
        self.angle_chars = {
            0: '│',     # Vertical line
            45: '╱',    # Diagonal up-right
            90: '─',    # Horizontal line
            135: '╲',   # Diagonal down-right
        }

    def get_nearest_angle_char(self, angle: float) -> str:
        """
        Find the nearest predefined angle character.

        Args:
            angle (float): Angle in degrees.

        Returns:
            str: UTF-8 character representing the angle.
        """
        normalized_angle = angle % 180
        closest_angle = min(self.angle_chars.keys(), key=lambda x: abs(x - normalized_angle))
        return self.angle_chars[closest_angle]

    def calculate_kernel_average(self, image: np.ndarray, coords: list, kernel_size=(10, 10)) -> np.ndarray:
        """
        Divide the image and the mask into kernels and calculate the average color.

        Args:
            image (numpy.ndarray): Input image.
            coords (list): List of object coordinates.
            kernel_size (tuple): Size of each kernel.

        Returns:
            numpy.ndarray: Image with averaged colors per kernel where objects are present.
        """
        # Create a mask for all objects
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for coord in coords:
            polygon = coord.astype(np.int32)
            cv2.fillPoly(full_mask, [polygon], 255)

        # Prepare result image
        kernel_colored_image = np.zeros_like(image)
        h, w, _ = image.shape
        kernel_h, kernel_w = kernel_size

        # Divide the image and mask into kernels
        for y in range(0, h, kernel_h):
            for x in range(0, w, kernel_w):
                kernel_region = image[y:y+kernel_h, x:x+kernel_w]
                kernel_mask = full_mask[y:y+kernel_h, x:x+kernel_w]

                # Check overlap with the mask
                if np.any(kernel_mask):
                    # Calculate average color for the kernel
                    avg_color = np.mean(kernel_region[kernel_mask > 0], axis=0).astype(np.uint8)

                    # Apply average color to the kernel
                    kernel_colored_image[y:y+kernel_h, x:x+kernel_w] = avg_color

        return kernel_colored_image

    def interpolate_multi_coord_mask_contour(self, image: np.ndarray, coords: list, kernel_size=(10, 10)) -> np.ndarray:
        """
        Divide the mask into kernels and interpolate the mask contours.

        Args:
            image (numpy.ndarray): Original image.
            coords (list): List of object coordinates.
            kernel_size (tuple): Size of kernels for interpolation.

        Returns:
            numpy.ndarray: Interpolated mask with UTF-8 characters per kernel.
        """
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for coord in coords:
            polygon = coord.astype(np.int32)
            cv2.fillPoly(full_mask, [polygon], 255)

        h, w = full_mask.shape
        kernel_h, kernel_w = kernel_size
        interpolated_mask = np.zeros_like(full_mask)

        # Process the mask kernel by kernel
        for y in range(0, h, kernel_h):
            for x in range(0, w, kernel_w):
                kernel_region = full_mask[y:y+kernel_h, x:x+kernel_w]

                if np.any(kernel_region):
                    # Find contours in the kernel
                    contours, _ = cv2.findContours(kernel_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)

                        for i in range(len(smoothed_contour) - 1):
                            pt1 = smoothed_contour[i][0]
                            pt2 = smoothed_contour[i+1][0]
                            pt1 = (pt1[0] + x, pt1[1] + y)  # Adjust for kernel offset
                            pt2 = (pt2[0] + x, pt2[1] + y)

                            angle = math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
                            angle_char = self.get_nearest_angle_char(angle)

                            mid_x = (pt1[0] + pt2[0]) // 2
                            mid_y = (pt1[1] + pt2[1]) // 2

                            cv2.putText(interpolated_mask, angle_char,
                                        (mid_x, mid_y),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, 255, 1)

        return interpolated_mask


class ImageSegmentation:
    def __init__(self, model_path='./yolov11m-seg.pt'):
        self.model = YOLO(model_path)
        self.ascii_art = ASCIIArtGenerator()

    def process_image(self, image_path: str, kernel_size=(10, 10)) -> dict:
        """
        Process the input image using YOLO segmentation and generate outputs.

        Args:
            image_path (str): Path to the input image.
            kernel_size (tuple): Size of kernels for processing.

        Returns:
            dict: Dictionary containing processed results.
        """
        results = self.model(image_path)
        image = cv2.imread(image_path)
        masks = results[0].masks
        coords = masks.xy

        kernel_colored_image = self.ascii_art.calculate_kernel_average(image, coords, kernel_size)
        interpolated_contour = self.ascii_art.interpolate_multi_coord_mask_contour(image, coords, kernel_size)

        cv2.imwrite('kernel_colored_image.jpg', kernel_colored_image)
        cv2.imwrite('interpolated_contour.jpg', interpolated_contour)

        return {
            'original_image': image,
            'kernel_colored_image': kernel_colored_image,
            'interpolated_contour': interpolated_contour,
            'coords': coords
        }


def main():
    image_path = 'car.jpg'
    segmenter = ImageSegmentation()

    result = segmenter.process_image(image_path, kernel_size=(10, 10))
    print("Image processed successfully")


if __name__ == '__main__':
    main()
