//! Configuration types for the OCR server and CLI.

use std::path::PathBuf;

/// Configuration for OCR processing
#[derive(Clone)]
pub struct OcrConfig {
    pub det_model: PathBuf,
    pub rec_model: PathBuf,
    pub dict_path: PathBuf,
    pub device: String,
}

/// Configuration for the HTTP server
#[derive(Clone)]
#[allow(dead_code)]
pub struct ServerConfig {
    pub ocr: OcrConfig,
    pub host: String,
    pub port: u16,
    pub workers: Option<usize>,
}
