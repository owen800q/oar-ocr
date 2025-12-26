//! PDF processing module for converting PDF pages to images.

use image::RgbImage;
use pdfium_render::prelude::*;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PdfError {
    #[error("Failed to initialize PDFium: {0}")]
    InitError(String),

    #[error("Failed to load PDF: {0}")]
    LoadError(String),

    #[error("Failed to render page {page}: {message}")]
    RenderError { page: usize, message: String },

    #[error("PDF has no pages")]
    EmptyPdf,
}

/// Configuration for PDF rendering
#[derive(Clone)]
pub struct PdfRenderSettings {
    /// DPI for rendering (default: 200)
    pub dpi: f32,
    /// Maximum dimension for rendered images (default: 4000)
    pub max_dimension: u32,
}

impl Default for PdfRenderSettings {
    fn default() -> Self {
        Self {
            dpi: 200.0,
            max_dimension: 4000,
        }
    }
}

/// PDF processor for converting PDF pages to images
pub struct PdfProcessor {
    pdfium: Pdfium,
    config: PdfRenderSettings,
}

impl PdfProcessor {
    /// Create a new PDF processor
    pub fn new(config: PdfRenderSettings) -> Result<Self, PdfError> {
        // Try to bind to system PDFium or use bundled one
        let pdfium = Pdfium::new(
            Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("./"))
                .or_else(|_| Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("/usr/lib")))
                .or_else(|_| Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("/usr/local/lib")))
                .or_else(|_| Pdfium::bind_to_library(Pdfium::pdfium_platform_library_name_at_path("/opt/homebrew/lib")))
                .or_else(|_| Pdfium::bind_to_system_library())
                .map_err(|e| PdfError::InitError(format!("Could not find PDFium library: {}", e)))?,
        );

        Ok(Self { pdfium, config })
    }

    /// Create a new PDF processor with default configuration
    pub fn new_default() -> Result<Self, PdfError> {
        Self::new(PdfRenderSettings::default())
    }

    /// Load a PDF from bytes and render all pages to images
    pub fn render_pdf_bytes(&self, bytes: &[u8]) -> Result<Vec<RgbImage>, PdfError> {
        let document = self
            .pdfium
            .load_pdf_from_byte_slice(bytes, None)
            .map_err(|e| PdfError::LoadError(e.to_string()))?;

        self.render_document(&document)
    }

    /// Load a PDF from a file path and render all pages to images
    pub fn render_pdf_file(&self, path: &Path) -> Result<Vec<RgbImage>, PdfError> {
        let document = self
            .pdfium
            .load_pdf_from_file(path, None)
            .map_err(|e| PdfError::LoadError(e.to_string()))?;

        self.render_document(&document)
    }

    /// Render all pages of a PDF document to images
    fn render_document(&self, document: &PdfDocument) -> Result<Vec<RgbImage>, PdfError> {
        let page_count = document.pages().len();

        if page_count == 0 {
            return Err(PdfError::EmptyPdf);
        }

        let mut images = Vec::with_capacity(page_count as usize);

        for (index, page) in document.pages().iter().enumerate() {
            let image = self
                .render_page(&page, index)
                .map_err(|e| PdfError::RenderError {
                    page: index + 1,
                    message: e.to_string(),
                })?;
            images.push(image);
        }

        Ok(images)
    }

    /// Render a single page to an RGB image
    fn render_page(&self, page: &PdfPage, _index: usize) -> Result<RgbImage, PdfError> {
        // Calculate render size based on DPI
        let width_points = page.width().value;
        let height_points = page.height().value;

        // Convert points to pixels at specified DPI (72 points per inch)
        let scale = self.config.dpi / 72.0;
        let mut width_px = (width_points * scale) as u32;
        let mut height_px = (height_points * scale) as u32;

        // Limit maximum dimension
        if width_px > self.config.max_dimension || height_px > self.config.max_dimension {
            let ratio = if width_px > height_px {
                self.config.max_dimension as f32 / width_px as f32
            } else {
                self.config.max_dimension as f32 / height_px as f32
            };
            width_px = (width_px as f32 * ratio) as u32;
            height_px = (height_px as f32 * ratio) as u32;
        }

        // Render the page
        let render_config = PdfRenderConfig::new()
            .set_target_width(width_px as i32)
            .set_target_height(height_px as i32)
            .render_form_data(true)
            .render_annotations(true);

        let bitmap = page
            .render_with_config(&render_config)
            .map_err(|e| PdfError::RenderError {
                page: 0,
                message: e.to_string(),
            })?;

        // Convert to DynamicImage then to RgbImage
        let dynamic_image = bitmap
            .as_image();

        Ok(dynamic_image.to_rgb8())
    }

    /// Get the number of pages in a PDF from bytes
    #[allow(dead_code)]
    pub fn page_count_from_bytes(&self, bytes: &[u8]) -> Result<usize, PdfError> {
        let document = self
            .pdfium
            .load_pdf_from_byte_slice(bytes, None)
            .map_err(|e| PdfError::LoadError(e.to_string()))?;

        Ok(document.pages().len() as usize)
    }
}

/// Check if bytes represent a PDF file (magic bytes: %PDF)
pub fn is_pdf_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[0..4] == b"%PDF"
}

/// Check if a file path has a PDF extension
pub fn is_pdf_path(path: &Path) -> bool {
    path.extension()
        .map(|ext| ext.to_ascii_lowercase() == "pdf")
        .unwrap_or(false)
}

/// Check if a URL points to a PDF (by extension or content-type hint)
pub fn is_pdf_url(url: &str) -> bool {
    let lower = url.to_lowercase();
    lower.ends_with(".pdf") || lower.contains(".pdf?") || lower.contains(".pdf#")
}
