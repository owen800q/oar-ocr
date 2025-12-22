//! CLI mode for OCR processing.

use crate::config::OcrConfig;
use crate::ocr::{download_image, load_image_from_path, OcrEngine, OcrError};
use std::path::Path;
use std::time::Instant;
use tracing::info;

/// Process an image from a URL
pub async fn process_url(
    url: &str,
    config: &OcrConfig,
    output_format: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();

    info!("Downloading image from URL...");
    let image = download_image(url).await?;
    let download_time = start.elapsed();
    info!("Downloaded in {:.2}ms", download_time.as_secs_f64() * 1000.0);

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

    Ok(())
}

/// Process an image from a local file
pub fn process_file(
    path: &Path,
    config: &OcrConfig,
    output_format: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let start = Instant::now();

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
