import math
import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple

class RealWorldDegradations:
    """
    Simulates real-world outdoor birding optical degradations:
    1. low_light: Gamma darkening + sensor Gaussian noise.
    2. motion_blur: Directional kernel blur simulating handheld camera shake.
    3. foliage_occlusion: Random organic green/brown leaf-like patches occluding 25-35% of the frame.
    4. distant_bird: Downscaling and padding into an expansive background so bird occupies <10% area.
    """

    @staticmethod
    def low_light(img: Image.Image, gamma: float = 2.4, noise_sigma: float = 16.0) -> Image.Image:
        """Simulate low light (underexposure + high ISO sensor shot noise)."""
        arr = np.array(img, dtype=np.float32) / 255.0
        # Gamma underexposure curve
        darkened = np.power(arr, gamma)
        # Gaussian ISO noise
        noise = np.random.normal(0, noise_sigma / 255.0, darkened.shape)
        noisy = np.clip(darkened + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).astype(np.uint8))

    @staticmethod
    def motion_blur(img: Image.Image, kernel_size: int = 15, angle: float = 45.0) -> Image.Image:
        """Simulate camera shake / motion blur with directional 2D convolution kernel."""
        # Create directional motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        rad = math.radians(angle)
        cos_val = math.cos(rad)
        sin_val = math.sin(rad)

        for i in range(-center, center + 1):
            x = int(round(center + i * cos_val))
            y = int(round(center + i * sin_val))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0

        if kernel.sum() == 0:
            kernel[center, center] = 1.0
        else:
            kernel /= kernel.sum()

        # Apply convolution across RGB channels
        arr = np.array(img, dtype=np.float32)
        blurred = np.zeros_like(arr)
        
        # Simple separable / 2D spatial padding
        pad = center
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        for c in range(3):
            for i in range(kernel_size):
                for j in range(kernel_size):
                    if kernel[i, j] > 0:
                        blurred[:, :, c] += kernel[i, j] * padded[i:i + arr.shape[0], j:j + arr.shape[1], c]

        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))

    @staticmethod
    def foliage_occlusion(img: Image.Image, occlusion_range: Tuple[float, float] = (0.25, 0.35)) -> Image.Image:
        """Simulate leaves and twigs occluding the bird (25-35% coverage)."""
        w, h = img.size
        arr = np.array(img).copy()
        target_occlusion_pct = np.random.uniform(occlusion_range[0], occlusion_range[1])
        target_pixels = int(w * h * target_occlusion_pct)

        covered = 0
        palette = [
            (34, 60, 25),    # dark olive green
            (48, 85, 34),    # deep foliage green
            (85, 107, 47),   # olive drab
            (101, 67, 33),   # twig brown
            (60, 42, 20),    # dark bark brown
        ]

        # Draw organic leaf-like elliptical splatters
        while covered < target_pixels:
            patch_w = np.random.randint(int(w * 0.12), int(w * 0.30))
            patch_h = np.random.randint(int(h * 0.12), int(h * 0.30))
            cx = np.random.randint(0, w)
            cy = np.random.randint(0, h)
            color = palette[np.random.randint(0, len(palette))]

            # Generate random ellipse mask
            y_indices, x_indices = np.ogrid[:h, :w]
            ellipse_mask = (((x_indices - cx) ** 2) / (patch_w ** 2) + 
                            ((y_indices - cy) ** 2) / (patch_h ** 2)) <= 1.0

            arr[ellipse_mask] = color
            covered += np.sum(ellipse_mask)

        # Soft edge blend
        leaf_img = Image.fromarray(arr)
        return leaf_img

    @staticmethod
    def distant_bird(img: Image.Image, bird_area_ratio: float = 0.08) -> Image.Image:
        """
        Simulate a telephoto shot where the bird occupies <10% of the frame,
        surrounded by wide bokeh / natural background texture.
        """
        w, h = img.size
        # Calculate target dimensions for bird crop
        scale = math.sqrt(bird_area_ratio) # e.g. sqrt(0.08) ~= 0.28
        new_w = max(16, int(w * scale))
        new_h = max(16, int(h * scale))

        small_bird = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Create naturalistic blurred background from the original image
        bg = img.resize((w, h), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=15))
        # Add subtle ambient noise to background
        bg_arr = np.array(bg, dtype=np.float32)
        noise = np.random.normal(0, 5, bg_arr.shape)
        bg = Image.fromarray(np.clip(bg_arr + noise, 0, 255).astype(np.uint8))

        # Paste small bird at a random realistic location (e.g. center or rule of thirds)
        max_x = w - new_w
        max_y = h - new_h
        paste_x = np.random.randint(int(max_x * 0.2), max(int(max_x * 0.2) + 1, int(max_x * 0.8)))
        paste_y = np.random.randint(int(max_y * 0.2), max(int(max_y * 0.2) + 1, int(max_y * 0.8)))

        bg.paste(small_bird, (paste_x, paste_y))
        return bg

    @classmethod
    def apply_degradation(cls, img: Image.Image, mode: str) -> Image.Image:
        if mode == "low_light":
            return cls.low_light(img)
        elif mode == "motion_blur":
            return cls.motion_blur(img)
        elif mode == "foliage_occlusion":
            return cls.foliage_occlusion(img)
        elif mode == "distant_bird":
            return cls.distant_bird(img)
        elif mode == "clean" or mode == "none":
            return img
        else:
            raise ValueError(f"Unknown degradation mode: {mode}")
