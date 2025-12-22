//! OCR processing logic shared between CLI and server modes.

use crate::config::OcrConfig;
use image::RgbImage;
#[cfg(feature = "cuda")]
use oar_ocr::core::config::OrtExecutionProvider;
use oar_ocr::core::config::OrtSessionConfig;
use oar_ocr::oarocr::{OAROCRBuilder, OAROCRResult, OAROCR};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OcrError {
    #[error("Failed to load image: {0}")]
    ImageLoad(String),

    #[error("Failed to download image: {0}")]
    Download(String),

    #[error("OCR processing failed: {0}")]
    Processing(String),

    #[error("Invalid configuration: {0}")]
    Config(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

/// Request to perform OCR on an image
#[derive(Debug, Deserialize)]
pub struct OcrRequest {
    /// URL of the image to process
    pub url: String,
}

/// A single text region detected in the image
#[derive(Debug, Serialize)]
pub struct TextRegionResponse {
    pub text: String,
    pub confidence: f32,
    pub bounding_box: BoundingBoxResponse,
}

/// Bounding box coordinates
#[derive(Debug, Serialize)]
pub struct BoundingBoxResponse {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// Response from OCR processing
#[derive(Debug, Serialize)]
pub struct OcrResponse {
    pub success: bool,
    pub text: String,
    pub regions: Vec<TextRegionResponse>,
    pub image_width: u32,
    pub image_height: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processing_time_ms: Option<f64>,
}

impl OcrResponse {
    pub fn error(message: String) -> Self {
        Self {
            success: false,
            text: String::new(),
            regions: Vec::new(),
            image_width: 0,
            image_height: 0,
            error: Some(message),
            processing_time_ms: None,
        }
    }
}

/// OCR Engine wrapper for thread-safe access
pub struct OcrEngine {
    ocr: OAROCR,
}

impl OcrEngine {
    /// Create a new OCR engine with the given configuration
    pub fn new(config: &OcrConfig) -> Result<Self, OcrError> {
        // Validate model files exist
        if !config.det_model.exists() {
            return Err(OcrError::ModelNotFound(format!(
                "Detection model not found: {}",
                config.det_model.display()
            )));
        }
        if !config.rec_model.exists() {
            return Err(OcrError::ModelNotFound(format!(
                "Recognition model not found: {}",
                config.rec_model.display()
            )));
        }
        if !config.dict_path.exists() {
            return Err(OcrError::ModelNotFound(format!(
                "Dictionary file not found: {}",
                config.dict_path.display()
            )));
        }

        let ort_config = parse_device_config(&config.device)?;

        let mut builder =
            OAROCRBuilder::new(&config.det_model, &config.rec_model, &config.dict_path);

        if let Some(config) = ort_config {
            builder = builder.ort_session(config);
        }

        let ocr = builder
            .build()
            .map_err(|e| OcrError::Config(e.to_string()))?;

        Ok(Self { ocr })
    }

    /// Process an image and return OCR results
    pub fn process(&self, image: RgbImage) -> Result<OAROCRResult, OcrError> {
        let results = self
            .ocr
            .predict(vec![image])
            .map_err(|e| OcrError::Processing(e.to_string()))?;

        results
            .into_iter()
            .next()
            .ok_or_else(|| OcrError::Processing("No results returned".to_string()))
    }

    /// Convert internal result to API response
    pub fn result_to_response(result: &OAROCRResult, processing_time_ms: f64) -> OcrResponse {
        let regions: Vec<TextRegionResponse> = result
            .text_regions
            .iter()
            .map(|region| {
                let bbox = &region.bounding_box;
                TextRegionResponse {
                    text: region
                        .text
                        .as_ref()
                        .map(|t| t.to_string())
                        .unwrap_or_default(),
                    confidence: region.confidence.unwrap_or(0.0),
                    bounding_box: BoundingBoxResponse {
                        x_min: bbox.x_min(),
                        y_min: bbox.y_min(),
                        x_max: bbox.x_max(),
                        y_max: bbox.y_max(),
                    },
                }
            })
            .collect();

        OcrResponse {
            success: true,
            text: result.concatenated_text("\n"),
            regions,
            image_width: result.input_img.width(),
            image_height: result.input_img.height(),
            error: None,
            processing_time_ms: Some(processing_time_ms),
        }
    }
}

/// Download an image from a URL
pub async fn download_image(url: &str) -> Result<RgbImage, OcrError> {
    let response = reqwest::get(url)
        .await
        .map_err(|e| OcrError::Download(format!("Failed to fetch URL: {}", e)))?;

    if !response.status().is_success() {
        return Err(OcrError::Download(format!(
            "HTTP error: {}",
            response.status()
        )));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| OcrError::Download(format!("Failed to read response body: {}", e)))?;

    load_image_from_bytes(&bytes)
}

/// Load an image from bytes
pub fn load_image_from_bytes(bytes: &[u8]) -> Result<RgbImage, OcrError> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| OcrError::ImageLoad(format!("Failed to decode image: {}", e)))?;

    Ok(img.to_rgb8())
}

/// Load an image from a file path
pub fn load_image_from_path(path: &std::path::Path) -> Result<RgbImage, OcrError> {
    let img = image::open(path)
        .map_err(|e| OcrError::ImageLoad(format!("Failed to load image: {}", e)))?;

    Ok(img.to_rgb8())
}

/// Parse device string and create OrtSessionConfig
fn parse_device_config(device: &str) -> Result<Option<OrtSessionConfig>, OcrError> {
    let device_lower = device.to_lowercase();

    if device_lower == "cpu" {
        return Ok(None);
    }

    #[cfg(feature = "cuda")]
    {
        if device_lower.starts_with("cuda") {
            let device_id = if device_lower == "cuda" {
                0
            } else if let Some(id_str) = device_lower.strip_prefix("cuda:") {
                id_str
                    .parse::<i32>()
                    .map_err(|_| OcrError::Config(format!("Invalid CUDA device ID: {}", device)))?
            } else {
                return Err(OcrError::Config(format!(
                    "Invalid device format: {}. Expected 'cuda' or 'cuda:N'",
                    device
                )));
            };

            let config = OrtSessionConfig::new().with_execution_providers(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(device_id),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    do_copy_in_default_stream: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ]);

            return Ok(Some(config));
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        if device_lower.starts_with("cuda") {
            return Err(OcrError::Config(format!(
                "CUDA device '{}' requested but CUDA feature is not enabled",
                device
            )));
        }
    }

    Err(OcrError::Config(format!("Unsupported device: {}", device)))
}

/// Thread-safe OCR engine wrapped in Arc
pub type SharedOcrEngine = Arc<OcrEngine>;
