from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import numpy as np

from ..utils.florence_utils import load_florence, run_florence_od_batch, parse_od_output, enable_trt_vision
from ..trackers.sort import Sort as BaseSort
from .system_service import get_configuration


logger = logging.getLogger(__name__)


class DetectorService:
    """Service for handling Florence-2 object detection operations."""
    
    def __init__(self):
        self._processor = None
        self._model = None
        self._device = None
        self._is_loaded = False
        self._model_id = None
        self._dtype_mode = None
        self._hf_token = None
    
    def is_loaded(self) -> bool:
        """Check if the detector model is loaded."""
        return self._is_loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self._is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_id": self._model_id,
            "device": self._device,
            "dtype_mode": self._dtype_mode
        }
    
    def load_model(self, model_id: str, dtype_mode: str, hf_token: Optional[str] = None) -> Tuple[Any, Any, str]:
        """Load the Florence-2 model for detection.
        
        Args:
            model_id: Florence-2 model identifier
            dtype_mode: Data type mode for the model
            hf_token: Hugging Face token for authentication
            
        Returns:
            Tuple of (processor, model, device)
        """
        logger.info(f"Loading Florence-2 model: {model_id}")
        
        self._processor, self._model, self._device = load_florence(
            model_id=model_id, 
            dtype_mode=dtype_mode, 
            hf_token=hf_token
        )
        
        self._is_loaded = True
        self._model_id = model_id
        self._dtype_mode = dtype_mode
        self._hf_token = hf_token
        
        logger.info(f"Florence-2 model {model_id} loaded on device {self._device}")
        return self._processor, self._model, self._device
    
    def enable_trt_vision(self, vision_h: int = 768, vision_w: int = 768, cache_dir: Optional[Path] = None) -> bool:
        """Enable TensorRT vision optimization.
        
        Args:
            vision_h: Vision height for TRT optimization
            vision_w: Vision width for TRT optimization
            cache_dir: Directory to cache TRT engines
            
        Returns:
            True if TRT was enabled successfully, False otherwise
        """
        if not self._is_loaded:
            logger.warning("Cannot enable TRT: model not loaded")
            return False
        
        try:
            if cache_dir is None:
                cache_dir = Path.home() / ".florence_trt_cache"
            
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            ok_trt = enable_trt_vision(
                self._processor,
                self._model,
                vision_h,
                vision_w,
                fp16=True,
                opset=17,
                cache_dir=str(cache_dir)
            )
            
            if ok_trt:
                logger.info("TRT vision optimization enabled successfully")
            else:
                logger.warning("TRT vision optimization failed")
            
            return ok_trt
            
        except Exception as e:
            logger.error(f"Failed to enable TRT vision: {e}")
            return False
    
    def detect_batch(self, images: List[Image.Image], prompts: List[str], max_new_tokens: int = 256) -> List[Any]:
        """Run object detection on a batch of images.
        
        Args:
            images: List of PIL Images to process
            prompts: List of prompts for each image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of detection outputs
        """
        if not self._is_loaded:
            raise RuntimeError("Detector model not loaded")
        
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        return run_florence_od_batch(
            self._model, 
            self._processor, 
            images, 
            device=self._device, 
            max_new_tokens=max_new_tokens, 
            prompts=prompts
        )
    
    def detect_single(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> Any:
        """Run object detection on a single image.
        
        Args:
            image: PIL Image to process
            prompt: Prompt for the image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Detection output
        """
        return self.detect_batch([image], [prompt], max_new_tokens)[0]
    
    def parse_detection_output(self, detection_output: Any) -> Tuple[List[List[float]], List[str], List[float]]:
        """Parse detection output to extract bounding boxes, labels, and scores.
        
        Args:
            detection_output: Raw detection output from Florence-2
            
        Returns:
            Tuple of (bounding_boxes, labels, scores)
        """
        return parse_od_output(detection_output)
    
    def detect_and_parse(self, images: List[Image.Image], prompts: List[str], max_new_tokens: int = 256) -> List[Tuple[List[List[float]], List[str], List[float]]]:
        """Run detection and parse outputs in one call.
        
        Args:
            images: List of PIL Images to process
            prompts: List of prompts for each image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            List of tuples containing (bounding_boxes, labels, scores) for each image
        """
        outputs = self.detect_batch(images, prompts, max_new_tokens)
        return [self.parse_detection_output(output) for output in outputs]
    
    def detect_single_and_parse(self, image: Image.Image, prompt: str, max_new_tokens: int = 256) -> Tuple[List[List[float]], List[str], List[float]]:
        """Run detection on a single image and parse the output.
        
        Args:
            image: PIL Image to process
            prompt: Prompt for the image
            max_new_tokens: Maximum number of new tokens to generate
            
        Returns:
            Tuple of (bounding_boxes, labels, scores)
        """
        output = self.detect_single(image, prompt, max_new_tokens)
        return self.parse_detection_output(output)
    
    def unload(self) -> None:
        """Unload the model and free resources."""
        if self._is_loaded:
            logger.info("Unloading Florence-2 model")
            self._processor = None
            self._model = None
            self._device = None
            self._is_loaded = False
            self._model_id = None
            self._dtype_mode = None
            self._hf_token = None


# Global detector service instance
_detector_service = DetectorService()


def get_detector_service() -> DetectorService:
    """Get the global detector service instance."""
    return _detector_service


def load_detector_model(model_id: str, dtype_mode: str, hf_token: Optional[str] = None) -> Tuple[Any, Any, str]:
    """Load the detector model using the global service.
    
    Args:
        model_id: Florence-2 model identifier
        dtype_mode: Data type mode for the model
        hf_token: Hugging Face token for authentication
        
    Returns:
        Tuple of (processor, model, device)
    """
    return _detector_service.load_model(model_id, dtype_mode, hf_token)


def is_detector_loaded() -> bool:
    """Check if the detector model is loaded."""
    return _detector_service.is_loaded()


def get_detector_info() -> Dict[str, Any]:
    """Get information about the loaded detector model."""
    return _detector_service.get_model_info()


def enable_detector_trt(vision_h: int = 768, vision_w: int = 768, cache_dir: Optional[Path] = None) -> bool:
    """Enable TensorRT optimization for the detector."""
    return _detector_service.enable_trt_vision(vision_h, vision_w, cache_dir)


def detect_batch(images: List[Image.Image], prompts: List[str], max_new_tokens: int = 256) -> List[Any]:
    """Run object detection on a batch of images."""
    return _detector_service.detect_batch(images, prompts, max_new_tokens)


def detect_single(image: Image.Image, prompt: str, max_new_tokens: int = 256) -> Any:
    """Run object detection on a single image."""
    return _detector_service.detect_single(image, prompt, max_new_tokens)


def parse_detection_output(detection_output: Any) -> Tuple[List[List[float]], List[str], List[float]]:
    """Parse detection output to extract bounding boxes, labels, and scores."""
    return _detector_service.parse_detection_output(detection_output)


def detect_and_parse(images: List[Image.Image], prompts: List[str], max_new_tokens: int = 256) -> List[Tuple[List[List[float]], List[str], List[float]]]:
    """Run detection and parse outputs in one call."""
    return _detector_service.detect_and_parse(images, prompts, max_new_tokens)


def detect_single_and_parse(image: Image.Image, prompt: str, max_new_tokens: int = 256) -> Tuple[List[List[float]], List[str], List[float]]:
    """Run detection on a single image and parse the output."""
    return _detector_service.detect_single_and_parse(image, prompt, max_new_tokens)


def unload_detector() -> None:
    """Unload the detector model and free resources."""
    _detector_service.unload()
