import random
import numpy as np
import torch
import logging
from scipy import signal
from scipy.ndimage import shift

logger = logging.getLogger(__name__)

def load_augmentations_from_config(config):

    AUGMENTATION_CLASSES = {
        "RandomAmplitudeScale": RandomAmplitudeScale,
        "RandomDCShift": RandomDCShift,
        "RandomTimeShift": RandomTimeShift,
        "RandomZeroMasking": RandomZeroMasking,
        "RandomAdditiveGaussianNoise": RandomAdditiveGaussianNoise,
        "RandomBandStopFilter": RandomBandStopFilter,
        "TimeWarping": TimeWarping,
        "TimeReverse": TimeReverse,
        "Permutation": Permutation,
        "CutoutResize": CutoutResize,
        "TailoredMixup": TailoredMixup,
        "AverageFilter": AverageFilter,
        "SignFlip": SignFlip
    }

    augmentations = []
    for aug_name, aug_params in config.get("augmentations", {}).items():
        if aug_name in AUGMENTATION_CLASSES:
            augmentation_class = AUGMENTATION_CLASSES[aug_name]
            augmentations.append(augmentation_class(**aug_params))
        else:
            logging.warning(f"Augmentation '{aug_name}' not recognized. Skipping.")
    return augmentations

class BaseAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        self.requires_x_random = False

    def should_apply(self):
        return torch.rand(1) < self.p

class RandomAmplitudeScale(BaseAugmentation):
    def __init__(self, range=(0.5, 2.0), p=0.5):
        super().__init__(p)
        self.range = range
        logger.debug(f"RandomAmplitudeScale range: {range}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                scale = random.uniform(self.range[0], self.range[1])
                logger.debug(f"Scaled by {scale}")
                return x * scale
        except Exception as e:
            logger.error(f"Error in RandomAmplitudeScale: {e}")
        return x

class RandomDCShift(BaseAugmentation):
    def __init__(self, range=(-10.0, 10.0), p=0.5):
        super().__init__(p)
        self.range = range
        logger.debug(f"RandomDCShift range: {range}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                shift_value = random.uniform(self.range[0], self.range[1])
                logger.debug(f"Shifted by {shift_value}")
                return x + shift_value
        except Exception as e:
            logger.error(f"Error in RandomDCShift: {e}")
        return x

class RandomTimeShift(BaseAugmentation):
    def __init__(self, range=(-300, 300), mode='constant', cval=0.0, p=0.5):
        super().__init__(p)
        self.range = range
        self.mode = mode
        self.cval = cval
        logger.debug(f"RandomTimeShift range: {range}, mode: {mode}, cval: {cval}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                t_shift = random.randint(self.range[0], self.range[1])
                x_shifted = shift(input=x, shift=t_shift, mode=self.mode, cval=self.cval)
                logger.debug(f"Time Shifted by : {t_shift}")
                return x_shifted
        except Exception as e:
            logger.error(f"Error in RandomTimeShift: {e}")
        return x

class RandomZeroMasking(BaseAugmentation):
    def __init__(self, range=(0, 300), p=0.5):
        super().__init__(p)
        self.range = range
        logger.debug(f"RandomZeroMasking range: {range}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                mask_len = random.randint(self.range[0], self.range[1])
                random_pos = random.randint(0, len(x) - mask_len)
                mask = np.ones_like(x)
                mask[random_pos:random_pos + mask_len] = 0
                logger.debug(f"Mask Length : {mask_len}, Position : {random_pos}")
                return x * mask
        except Exception as e:
            logger.error(f"Error in RandomZeroMasking: {e}")
        return x

class RandomAdditiveGaussianNoise(BaseAugmentation):
    def __init__(self, range=(0.0, 0.2), p=0.5):
        super().__init__(p)
        self.range = range
        logger.debug(f"RandomAdditiveGaussianNoise range: {range}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                sigma = random.uniform(self.range[0], self.range[1])
                noise = np.random.normal(0, sigma, x.shape)
                logger.debug(f"Gaussian Noise std = {sigma}")
                return x + noise
        except Exception as e:
            logger.error(f"Error in RandomAdditiveGaussianNoise: {e}")
        return x

class RandomBandStopFilter(BaseAugmentation):
    def __init__(self, range=(0.5, 30.0), band_width=2.0, sampling_rate=100.0, p=0.5):
        super().__init__(p)
        self.range = range
        self.band_width = band_width
        self.sampling_rate = sampling_rate
        logger.debug(f"RandomBandStopFilter range: {range}, band_width: {band_width}, sampling_rate: {sampling_rate}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                low_freq = random.uniform(self.range[0], self.range[1] - self.band_width)
                center_freq = low_freq + self.band_width / 2.0
                b, a = signal.iirnotch(center_freq, Q=center_freq / self.band_width, fs=self.sampling_rate)
                x_filtered = signal.lfilter(b, a, x)
                logger.debug(f"Central Freq : {center_freq}")
                return x_filtered
        except Exception as e:
            logger.error(f"Error in RandomBandStopFilter: {e}")
        return x

class TimeWarping(BaseAugmentation):
    def __init__(self, n_segments=4, scale_range=(0.5, 2.0), p=0.5):
        super().__init__(p)
        self.n_segments = n_segments
        self.scale_range = scale_range
        logger.debug(f"TimeWarping n_segments: {n_segments}, scale range: {scale_range}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                L = len(x)
                segment_length = L // self.n_segments
                segments = []
                for i in range(self.n_segments):
                    start = i * segment_length
                    end = start + segment_length if i < self.n_segments -1 else L
                    Si = x[start:end]
                    omega = random.uniform(self.scale_range[0], self.scale_range[1])
                    new_length = int(len(Si) * omega)
                    if new_length < 1:
                        new_length = 1
                    Si_transformed = signal.resample(Si, new_length)
                    segments.append(Si_transformed)
                x_aug = np.concatenate(segments)
                x_aug = signal.resample(x_aug, L)
                logger.debug(f"Time Warped with segments: {self.n_segments}, scale range: {self.scale_range}")
                return x_aug
        except Exception as e:
            logger.error(f"Error in TimeWarping: {e}")
        return x

class TimeReverse(BaseAugmentation):
    def __init__(self, p=0.5):
        super().__init__(p)
        logger.debug(f"TimeReverse p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                logger.debug("Time Reversed")
                return np.flip(x).copy()
        except Exception as e:
            logger.error(f"Error in TimeReverse: {e}")
        return x

class Permutation(BaseAugmentation):
    def __init__(self, n_segments=4, p=0.5):
        super().__init__(p)
        self.n_segments = n_segments
        logger.debug(f"Permutation n_segments: {n_segments}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                L = len(x)
                segment_length = L // self.n_segments
                segments = []
                indices = []
                for i in range(self.n_segments):
                    start = i * segment_length
                    end = start + segment_length if i < self.n_segments -1 else L
                    Si = x[start:end]
                    segments.append(Si)
                    indices.append(i)
                random.shuffle(indices)
                shuffled_segments = [segments[i] for i in indices]
                x_aug = np.concatenate(shuffled_segments)
                logger.debug(f"Permuted with segments: {self.n_segments}")
                return x_aug
        except Exception as e:
            logger.error(f"Error in Permutation: {e}")
        return x

class CutoutResize(BaseAugmentation):
    def __init__(self, n_segments=4, p=0.5):
        super().__init__(p)
        self.n_segments = n_segments
        logger.debug(f"CutoutResize n_segments: {n_segments}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                L = len(x)
                segment_length = L // self.n_segments
                segments = []
                for i in range(self.n_segments):
                    start = i * segment_length
                    end = start + segment_length if i < self.n_segments -1 else L
                    Si = x[start:end]
                    segments.append(Si)
                r = random.randint(0, self.n_segments - 1)
                del segments[r]
                x_aug = np.concatenate(segments)
                x_aug = signal.resample(x_aug, L)
                logger.debug(f"Cutout and Resized with segments: {self.n_segments}")
                return x_aug
        except Exception as e:
            logger.error(f"Error in CutoutResize: {e}")
        return x

class AverageFilter(BaseAugmentation):
    def __init__(self, k_range=(3, 10), p=0.5):
        super().__init__(p)
        self.k_range = k_range
        logger.debug(f"AverageFilter k_range: {k_range}, p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                k = random.randint(self.k_range[0], self.k_range[1])
                kernel = np.ones(k) / k
                x_aug = np.convolve(x, kernel, mode='same')
                logger.debug(f"Averaged with kernel size: {k}")
                return x_aug
        except Exception as e:
            logger.error(f"Error in AverageFilter: {e}")
        return x

class SignFlip(BaseAugmentation):
    def __init__(self, p=0.5):
        super().__init__(p)
        logger.debug(f"SignFlip p: {p}")

    def __call__(self, x, x_random=None):
        try:
            if self.should_apply():
                logger.debug("Sign Flipped")
                return (-x).copy()
        except Exception as e:
            logger.error(f"Error in SignFlip: {e}")
        return x

class TailoredMixup(BaseAugmentation):
    def __init__(self, p=0.5, fs=100, beta=0.5):
        super().__init__(p)
        self.fs = fs
        self.beta = beta
        self.requires_x_random = True
        logger.debug(f"TailoredMixup p: {p}, fs: {fs}, beta: {beta}")

    def __call__(self, x_anchor, x_random=None):
        try:
            if self.should_apply() and x_random is not None:
                X_anchor = np.fft.fft(x_anchor)
                X_random = np.fft.fft(x_random)
                A_anchor = np.abs(X_anchor)
                P_anchor = np.angle(X_anchor)
                A_random = np.abs(X_random)
                P_random = np.angle(X_random)
                lambda_A = random.uniform(self.beta, 1.0)
                lambda_P = random.uniform(self.beta, 1.0)
                A_mix = lambda_A * A_anchor + (1 - lambda_A) * A_random
                delta_theta = P_anchor - P_random
                delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
                P_mix = P_anchor - delta_theta * (1 - lambda_P)
                X_mix = A_mix * np.exp(1j * P_mix)
                x_aug = np.fft.ifft(X_mix).real
                logger.debug("Tailored Mixup applied")
                return x_aug
        except Exception as e:
            logger.error(f"Error in TailoredMixup: {e}")
        return x_anchor.copy()