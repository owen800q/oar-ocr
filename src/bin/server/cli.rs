//! CLI mode for OCR processing.

use crate::config::OcrConfig;
use crate::ocr::{download_bytes, load_image_from_path, OcrEngine, OcrError};
use crate::pdf::{is_pdf_bytes, is_pdf_path, is_pdf_url, PdfProcessor};
use std::path::Path;
use std::time::Instant;
use tracing::info;

/// Process a URL (image or PDF)
pub async fn process_url(
    url: &str,
    config: &OcrConfig,
    output_format: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();

    info!("Downloading content from URL...");
    let bytes = download_bytes(url).await?;
    let download_time = start.elapsed();
    info!("Downloaded {} bytes in {:.2}ms", bytes.len(), download_time.as_secs_f64() * 1000.0);

    // Check if it's a PDF
    if is_pdf_url(url) || is_pdf_bytes(&bytes) {
        info!("Detected PDF content, processing as multi-page document...");
        process_pdf_bytes(&bytes, config, output_format, start)?;
    } else {
        info!("Processing as image...");
        let image = crate::ocr::load_image_from_bytes(&bytes)?;

        info!("Initializing OCR engine...");
        let engine = OcrEngine::new(config)?;
        let init_time = start.elapsed() - download_time;
        info!("Engine initialized in {:.2}ms", init_time.as_secs_f64() * 1000.0);

        info!("Processing image ({}x{})...", image.width(), image.height());
        let ocr_start = Instant::now();
        let result = engine.process(image)?;
        let processing_time = ocr_start.elapsed();
        info!("OCR completed in {:.2}ms", processing_time.as_secs_f64() * 1000.0);

        output_result(&result, output_format, processing_time.as_secs_f64() * 1000.0)?;
    }

    Ok(())
}

/// Process a local file (image or PDF)
pub fn process_file(
    path: &Path,
    config: &OcrConfig,
    output_format: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();

    // Check if it's a PDF
    if is_pdf_path(path) {
        info!("Detected PDF file, processing as multi-page document...");
        process_pdf_file(path, config, output_format, start)?;
    } else {
        info!("Loading image from file...");
        let image = load_image_from_path(path)?;
        let load_time = start.elapsed();
        info!("Loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);

        info!("Initializing OCR engine...");
        let engine = OcrEngine::new(config)?;
        let init_time = start.elapsed() - load_time;
        info!("Engine initialized in {:.2}ms", init_time.as_secs_f64() * 1000.0);

        info!("Processing image ({}x{})...", image.width(), image.height());
        let ocr_start = Instant::now();
        let result = engine.process(image)?;
        let processing_time = ocr_start.elapsed();
        info!("OCR completed in {:.2}ms", processing_time.as_secs_f64() * 1000.0);

        output_result(&result, output_format, processing_time.as_secs_f64() * 1000.0)?;
    }

    Ok(())
}

/// Process PDF from bytes
fn process_pdf_bytes(
    bytes: &[u8],
    config: &OcrConfig,
    output_format: &str,
    start: Instant,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Initializing PDF processor...");
    let pdf_processor = PdfProcessor::new_default()
        .map_err(|e| OcrError::Pdf(e.to_string()))?;

    info!("Rendering PDF pages to images...");
    let images = pdf_processor.render_pdf_bytes(bytes)
        .map_err(|e| OcrError::Pdf(e.to_string()))?;
    let render_time = start.elapsed();
    info!("Rendered {} pages in {:.2}ms", images.len(), render_time.as_secs_f64() * 1000.0);

    process_pdf_images(images, config, output_format, start)
}

/// Process PDF from file path
fn process_pdf_file(
    path: &Path,
    config: &OcrConfig,
    output_format: &str,
    start: Instant,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Initializing PDF processor...");
    let pdf_processor = PdfProcessor::new_default()
        .map_err(|e| OcrError::Pdf(e.to_string()))?;

    info!("Rendering PDF pages to images...");
    let images = pdf_processor.render_pdf_file(path)
        .map_err(|e| OcrError::Pdf(e.to_string()))?;
    let render_time = start.elapsed();
    info!("Rendered {} pages in {:.2}ms", images.len(), render_time.as_secs_f64() * 1000.0);

    process_pdf_images(images, config, output_format, start)
}

/// Process rendered PDF images through OCR
fn process_pdf_images(
    images: Vec<image::RgbImage>,
    config: &OcrConfig,
    output_format: &str,
    start: Instant,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let page_count = images.len();

    info!("Initializing OCR engine...");
    let engine = OcrEngine::new(config)?;
    let init_time = start.elapsed();
    info!("Engine initialized in {:.2}ms", init_time.as_secs_f64() * 1000.0);

    info!("Processing {} PDF pages...", page_count);
    let ocr_start = Instant::now();
    let results = engine.process_multiple(images)?;
    let processing_time = ocr_start.elapsed();
    info!("OCR completed in {:.2}ms", processing_time.as_secs_f64() * 1000.0);

    output_multipage_result(&results, output_format, processing_time.as_secs_f64() * 1000.0)?;

    Ok(())
}

/// Output the OCR result in the specified format
fn output_result(
    result: &oar_ocr::oarocr::OAROCRResult,
    format: &str,
    processing_time_ms: f64,
) -> Result<(), OcrError> {
    match format {
        "json" => {
            let response = crate::ocr::OcrEngine::result_to_response(result, processing_time_ms);
            println!("{}", serde_json::to_string(&response).unwrap());
        }
        "text" => {
            println!("{}", result.concatenated_text("\n"));
        }
        "pretty" | _ => {
            println!("\n=== OCR Results ===");
            println!("Image size: {}x{}", result.input_img.width(), result.input_img.height());
            println!("Processing time: {:.2}ms", processing_time_ms);
            println!("Text regions: {}", result.text_regions.len());
            println!();

            if result.text_regions.is_empty() {
                println!("No text detected.");
            } else {
                println!("--- Detected Text ---");
                for (idx, region) in result.text_regions.iter().enumerate() {
                    let text = region
                        .text
                        .as_ref()
                        .map(|t| t.to_string())
                        .unwrap_or_else(|| "<no text>".to_string());
                    let confidence = region.confidence.unwrap_or(0.0) * 100.0;
                    let bbox = &region.bounding_box;

                    println!(
                        "[{}] \"{}\" ({:.1}%)",
                        idx + 1,
                        text,
                        confidence
                    );
                    println!(
                        "    Position: [{:.1}, {:.1}] - [{:.1}, {:.1}]",
                        bbox.x_min(),
                        bbox.y_min(),
                        bbox.x_max(),
                        bbox.y_max()
                    );
                }
                println!();
                println!("--- Full Text ---");
                println!("{}", result.concatenated_text("\n"));
            }
        }
    }

    Ok(())
}

/// Output multi-page OCR results in the specified format
fn output_multipage_result(
    results: &[oar_ocr::oarocr::OAROCRResult],
    format: &str,
    processing_time_ms: f64,
) -> Result<(), OcrError> {
    match format {
        "json" => {
            let response = crate::ocr::OcrEngine::results_to_multipage_response(results, processing_time_ms);
            println!("{}", serde_json::to_string(&response).unwrap());
        }
        "text" => {
            for (idx, result) in results.iter().enumerate() {
                if idx > 0 {
                    println!("\n--- Page {} ---\n", idx + 1);
                }
                println!("{}", result.concatenated_text("\n"));
            }
        }
        "pretty" | _ => {
            println!("\n=== PDF OCR Results ===");
            println!("Total pages: {}", results.len());
            println!("Processing time: {:.2}ms", processing_time_ms);
            println!();

            for (idx, result) in results.iter().enumerate() {
                println!("========== Page {} ==========", idx + 1);
                println!("Image size: {}x{}", result.input_img.width(), result.input_img.height());
                println!("Text regions: {}", result.text_regions.len());
                println!();

                if result.text_regions.is_empty() {
                    println!("No text detected on this page.");
                } else {
                    println!("--- Detected Text ---");
                    for (region_idx, region) in result.text_regions.iter().enumerate() {
                        let text = region
                            .text
                            .as_ref()
                            .map(|t| t.to_string())
                            .unwrap_or_else(|| "<no text>".to_string());
                        let confidence = region.confidence.unwrap_or(0.0) * 100.0;

                        println!(
                            "[{}] \"{}\" ({:.1}%)",
                            region_idx + 1,
                            text,
                            confidence
                        );
                    }
                    println!();
                    println!("--- Page Text ---");
                    println!("{}", result.concatenated_text("\n"));
                }
                println!();
            }

            // Print combined text at the end
            println!("========== Combined Text ==========");
            for (idx, result) in results.iter().enumerate() {
                if idx > 0 {
                    println!("\n--- Page {} ---\n", idx + 1);
                }
                println!("{}", result.concatenated_text("\n"));
            }
        }
    }

    Ok(())
}
