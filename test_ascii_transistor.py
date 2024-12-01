import pytest
import numpy as np
import os
import cv2

from ascii_transistor import ASCIIArtGenerator, ImageSegmentation

@pytest.fixture
def ascii_generator():
    """Fixture to create an instance of ASCIIArtGenerator for reuse across tests"""
    return ASCIIArtGenerator()

@pytest.fixture
def test_image_path():
    """Fixture to provide a test image path"""
    test_image_path = 'test_car.jpg'
    return test_image_path

class TestGrayScaleConversion:
    def test_input_validation(self, ascii_generator):
        """Test gray scale conversion with different input types"""
        # Test with None
        with pytest.raises(ValueError, match="Input image cannot be None"):
            ascii_generator.convert_to_grayscale(None)
        
        # Test with empty array
        with pytest.raises(ValueError, match="Input image is empty"):
            ascii_generator.convert_to_grayscale(np.array([]))

    def test_grayscale_output_properties(self, ascii_generator):
        """Test properties of grayscale converted image"""
        # Create a test color image
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Convert to grayscale
        gray_image = ascii_generator.convert_to_grayscale(test_image)
        
        # Assertions
        assert gray_image.ndim == 2, "Grayscale image should be 2D"
        assert gray_image.dtype == np.uint8, "Grayscale image should be uint8"
        assert gray_image.shape == test_image.shape[:2], "Grayscale image should have same height and width"

class TestASCIIMapping:
    def test_grayscale_to_ascii_mapping(self, ascii_generator):
        """Test ASCII character mapping for different grayscale intensities"""
        test_cases = [
            (0, '@'),     # Darkest
            (64, '#'),    # Dark
            (128, '='),   # Medium
            (192, '.'),   # Light
            (255, ' ')    # Brightest
        ]
        
        for intensity, expected_char in test_cases:
            result = ascii_generator.grayscale_to_ascii(intensity)
            assert result == expected_char, f"Failed for intensity {intensity}"

    def test_ascii_mapping_edge_cases(self, ascii_generator):
        """Test edge cases in ASCII mapping"""
        # Test values outside expected range
        with pytest.raises(ValueError, match="Grayscale intensity must be between 0 and 255"):
            ascii_generator.grayscale_to_ascii(-1)
        
        with pytest.raises(ValueError, match="Grayscale intensity must be between 0 and 255"):
            ascii_generator.grayscale_to_ascii(256)

class TestContourMapping:
    def test_get_nearest_angle_char(self, ascii_generator):
        """Test character mapping for different angles"""
        test_cases = [
            (0, '│'),    # Vertical line
            (45, '╱'),   # Diagonal up-right
            (90, '─'),   # Horizontal line
            (135, '╲'),  # Diagonal down-right
            (20, '│'),   # Close to vertical
            (60, '╱'),   # Close to diagonal up-right
            (100, '─'),  # Close to horizontal
            (160, '╲')   # Close to diagonal down-right
        ]

        for angle, expected_char in test_cases:
            result = ascii_generator.get_nearest_angle_char(angle)
            assert result == expected_char, f"Failed for angle {angle}"

    def test_contour_character_exclusion(self, ascii_generator):
        """Ensure ASCII mapping doesn't apply to contour pixels"""
        # Create a test grayscale image with a contour
        test_image = np.zeros((100, 100), dtype=np.uint8)
        test_image[25:75, 25:75] = 128  # Mid-intensity
        
        # Simulate contour
        contour = np.array([[25, 25], [75, 25], [75, 75], [25, 75]])
        
        # Get the character for a contour pixel
        contour_pixel_intensity = test_image[25, 25]
        result = ascii_generator.map_pixel(contour_pixel_intensity, is_contour=True)
        
        # Contour pixels should be mapped differently
        assert result == '│', "Contour pixel should be mapped to a specific line character"

class TestImageProcessing:
    def test_kernel_average_calculation(self, ascii_generator):
        """Test kernel average calculation"""
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 0, 0]  # Red square
        
        # Simulate object coordinates
        test_coords = [np.array([[25, 25], [75, 25], [75, 75], [25, 75]])]
        
        # Calculate kernel average
        result = ascii_generator.calculate_kernel_average(
            test_image, 
            test_coords, 
            kernel_size=(25, 25)
        )
        
        # Validate result
        assert result is not None, "Kernel average calculation failed"
        assert result.shape == test_image.shape, "Result shape should match input image"
        
        # Check if the masked region maintains color characteristics
        masked_region = result[25:75, 25:75]
        assert np.any(masked_region), "Masked region should not be empty"

class TestImageSegmentation:
    def test_image_segmentation_process(self, test_image_path):
        """Comprehensive test for image segmentation process"""
        segmenter = ImageSegmentation()
        
        # Process the image
        result = segmenter.process_image(test_image_path, kernel_size=(10, 10))
        
        # Validate result keys
        expected_keys = [
            'original_image', 
            'kernel_colored_image', 
            'interpolated_contour', 
            'coords'
        ]
        for key in expected_keys:
            assert key in result, f"{key} is missing from segmentation result"
        
        # Validate output image files
        output_files = [
            'kernel_colored_image.jpg', 
            'interpolated_contour.jpg'
        ]
        for file in output_files:
            assert os.path.exists(file), f"{file} was not generated"
        
        # Optional: Clean up generated files
        for file in output_files:
            os.remove(file)

# Additional tests can be added as needed