"""
Configuration management for Fire Detection System.

This module provides a centralized configuration management class that handles
all configuration parameters for the fire detection system, including model settings,
detection thresholds, logging configuration, and more.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Configuration for the fire detection model."""
    model_name: str = "yolov8m"
    model_path: str = "weights/fire_detection_model.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    device: str = "cuda"  # 'cuda', 'cpu', or specific GPU device
    half_precision: bool = True
    batch_size: int = 32
    input_size: int = 640


@dataclass
class DetectionConfig:
    """Configuration for fire detection parameters."""
    min_area_threshold: int = 50  # Minimum pixel area to consider as fire
    max_area_threshold: int = 500000  # Maximum pixel area
    flame_color_range: Dict[str, list] = field(default_factory=lambda: {
        "lower_hsv": [0, 100, 100],
        "upper_hsv": [20, 255, 255]
    })
    consecutive_frames: int = 3  # Frames to confirm detection
    temporal_smoothing: bool = True
    enable_edge_detection: bool = True


@dataclass
class AlertConfig:
    """Configuration for alert system."""
    alert_enabled: bool = True
    alert_threshold: float = 0.7
    email_enabled: bool = False
    email_recipients: list = field(default_factory=list)
    sms_enabled: bool = False
    sms_recipients: list = field(default_factory=list)
    webhook_enabled: bool = False
    webhook_url: str = ""
    alert_cooldown_seconds: int = 60


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    log_level: str = "INFO"
    log_file: str = "logs/fire_detection.log"
    max_log_size_mb: int = 10
    backup_count: int = 5
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True


@dataclass
class VideoConfig:
    """Configuration for video input/output."""
    input_source: str = "0"  # Default to webcam
    output_path: Optional[str] = "output/detection_output.mp4"
    save_output: bool = True
    fps: int = 30
    frame_width: int = 1280
    frame_height: int = 720
    enable_preprocessing: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    inference_optimization: bool = True
    use_quantization: bool = False
    num_threads: int = 4
    cache_size_mb: int = 512
    enable_profiling: bool = False
    max_queue_size: int = 100


class Config:
    """
    Main configuration management class for the Fire Detection System.
    
    This class handles loading, validating, and managing all configuration
    parameters for the fire detection system.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to JSON configuration file. If not provided,
                        uses default configurations.
        """
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-configurations
        self.model = ModelConfig()
        self.detection = DetectionConfig()
        self.alert = AlertConfig()
        self.logging = LoggingConfig()
        self.video = VideoConfig()
        self.performance = PerformanceConfig()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
        
        # Validate configuration
        self.validate()
    
    def load_from_file(self, config_file: str) -> None:
        """
        Load configuration from a JSON file.
        
        Args:
            config_file: Path to JSON configuration file.
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist.
            json.JSONDecodeError: If JSON is invalid.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            self._update_from_dict(config_data)
            self.logger.info(f"Configuration loaded from {config_file}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    def _update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration data.
        """
        if "model" in config_dict:
            self.model = ModelConfig(**config_dict["model"])
        if "detection" in config_dict:
            self.detection = DetectionConfig(**config_dict["detection"])
        if "alert" in config_dict:
            self.alert = AlertConfig(**config_dict["alert"])
        if "logging" in config_dict:
            self.logging = LoggingConfig(**config_dict["logging"])
        if "video" in config_dict:
            self.video = VideoConfig(**config_dict["video"])
        if "performance" in config_dict:
            self.performance = PerformanceConfig(**config_dict["performance"])
    
    def _load_from_env(self) -> None:
        """Load configuration overrides from environment variables."""
        # Model settings
        if env_val := os.getenv("FD_MODEL_NAME"):
            self.model.model_name = env_val
        if env_val := os.getenv("FD_CONFIDENCE_THRESHOLD"):
            self.model.confidence_threshold = float(env_val)
        if env_val := os.getenv("FD_DEVICE"):
            self.model.device = env_val
        
        # Detection settings
        if env_val := os.getenv("FD_MIN_AREA"):
            self.detection.min_area_threshold = int(env_val)
        
        # Alert settings
        if env_val := os.getenv("FD_ALERT_ENABLED"):
            self.alert.alert_enabled = env_val.lower() == "true"
        if env_val := os.getenv("FD_ALERT_THRESHOLD"):
            self.alert.alert_threshold = float(env_val)
        
        # Logging settings
        if env_val := os.getenv("FD_LOG_LEVEL"):
            self.logging.log_level = env_val
        
        # Video settings
        if env_val := os.getenv("FD_INPUT_SOURCE"):
            self.video.input_source = env_val
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid.
        """
        # Validate model config
        if not 0 <= self.model.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= self.model.iou_threshold <= 1:
            raise ValueError("iou_threshold must be between 0 and 1")
        if self.model.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        # Validate detection config
        if self.detection.min_area_threshold < 0:
            raise ValueError("min_area_threshold must be non-negative")
        if self.detection.consecutive_frames <= 0:
            raise ValueError("consecutive_frames must be positive")
        
        # Validate alert config
        if not 0 <= self.alert.alert_threshold <= 1:
            raise ValueError("alert_threshold must be between 0 and 1")
        if self.alert.alert_cooldown_seconds < 0:
            raise ValueError("alert_cooldown_seconds must be non-negative")
        
        # Validate video config
        if self.video.fps <= 0:
            raise ValueError("fps must be positive")
        if self.video.frame_width <= 0 or self.video.frame_height <= 0:
            raise ValueError("frame dimensions must be positive")
        
        self.logger.info("Configuration validation passed")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of all configuration.
        """
        return {
            "model": asdict(self.model),
            "detection": asdict(self.detection),
            "alert": asdict(self.alert),
            "logging": asdict(self.logging),
            "video": asdict(self.video),
            "performance": asdict(self.performance),
        }
    
    def save_to_file(self, output_file: str) -> None:
        """
        Save current configuration to a JSON file.
        
        Args:
            output_file: Path where to save the configuration file.
        """
        # Create directory if it doesn't exist
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        
        self.logger.info(f"Configuration saved to {output_file}")
    
    def get_model_config(self) -> ModelConfig:
        """Get model configuration."""
        return self.model
    
    def get_detection_config(self) -> DetectionConfig:
        """Get detection configuration."""
        return self.detection
    
    def get_alert_config(self) -> AlertConfig:
        """Get alert configuration."""
        return self.alert
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration."""
        return self.logging
    
    def get_video_config(self) -> VideoConfig:
        """Get video configuration."""
        return self.video
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        return self.performance
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(model={self.model}, detection={self.detection}, alert={self.alert})"


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_file: Path to configuration file (only used on first call).
        
    Returns:
        Global Config instance.
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
