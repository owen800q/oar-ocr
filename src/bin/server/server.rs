//! HTTP server for OCR processing.

use crate::config::ServerConfig;
use crate::ocr::{download_bytes, MultiPageOcrResponse, OcrEngine, OcrRequest, OcrResponse, SharedOcrEngine};
use crate::pdf::{is_pdf_bytes, is_pdf_url, PdfProcessor};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info};

/// Application state shared across handlers
struct AppState {
    engine: SharedOcrEngine,
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    pdf_support: bool,
}

/// Run the HTTP server
pub async fn run_server(
    config: ServerConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize OCR engine
    info!("Initializing OCR engine...");
    let engine = OcrEngine::new(&config.ocr)?;
    let engine = Arc::new(engine);
    info!("OCR engine initialized successfully");

    let state = Arc::new(AppState { engine });

    // Configure CORS
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build router
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/ocr", post(ocr_handler))
        .route("/api/v1/ocr", post(ocr_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Parse address
    let addr: SocketAddr = format!("{}:{}", config.host, config.port)
        .parse()
        .map_err(|e| format!("Invalid address: {}", e))?;

    info!("Server listening on http://{}", addr);
    info!("Endpoints:");
    info!("  GET  /health     - Health check");
    info!("  POST /ocr        - OCR processing (images and PDFs)");
    info!("  POST /api/v1/ocr - OCR processing (versioned API)");

    // Create listener
    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Run server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
    Ok(())
}

/// Health check endpoint
async fn health_handler() -> Json<HealthResponse> {
    // Check if PDFium is available
    let pdf_support = PdfProcessor::new_default().is_ok();

    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        pdf_support,
    })
}

/// OCR processing endpoint - handles both images and PDFs
async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OcrRequest>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4().to_string();
    info!(request_id = %request_id, url = %request.url, "Processing OCR request");

    let start = Instant::now();

    // Download content
    let bytes = match download_bytes(&request.url).await {
        Ok(b) => b,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "Failed to download content");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(OcrResponse::error(format!("Failed to download: {}", e))).unwrap()),
            );
        }
    };

    let download_time = start.elapsed();
    info!(
        request_id = %request_id,
        bytes = bytes.len(),
        download_ms = download_time.as_secs_f64() * 1000.0,
        "Content downloaded"
    );

    // Check if it's a PDF
    if is_pdf_url(&request.url) || is_pdf_bytes(&bytes) {
        info!(request_id = %request_id, "Detected PDF content, processing as multi-page document");
        return process_pdf_request(&request_id, &bytes, &state.engine, start).await;
    }

    // Process as image
    let image = match crate::ocr::load_image_from_bytes(&bytes) {
        Ok(img) => img,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "Failed to decode image");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(OcrResponse::error(format!("Failed to decode image: {}", e))).unwrap()),
            );
        }
    };

    info!(
        request_id = %request_id,
        width = image.width(),
        height = image.height(),
        "Image decoded"
    );

    // Process OCR
    let ocr_start = Instant::now();
    let result = match state.engine.process(image) {
        Ok(r) => r,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "OCR processing failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::to_value(OcrResponse::error(format!("OCR processing failed: {}", e))).unwrap()),
            );
        }
    };

    let processing_time = ocr_start.elapsed();
    let total_time = start.elapsed();

    info!(
        request_id = %request_id,
        regions = result.text_regions.len(),
        ocr_ms = processing_time.as_secs_f64() * 1000.0,
        total_ms = total_time.as_secs_f64() * 1000.0,
        "OCR completed"
    );

    let response = OcrEngine::result_to_response(&result, processing_time.as_secs_f64() * 1000.0);

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Process a PDF request
async fn process_pdf_request(
    request_id: &str,
    bytes: &[u8],
    engine: &OcrEngine,
    start: Instant,
) -> (StatusCode, Json<serde_json::Value>) {
    // Initialize PDF processor
    let pdf_processor = match PdfProcessor::new_default() {
        Ok(p) => p,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "Failed to initialize PDF processor");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::to_value(MultiPageOcrResponse::error(
                    format!("PDF processing not available: {}. Please install PDFium library.", e)
                )).unwrap()),
            );
        }
    };

    // Render PDF pages
    let images = match pdf_processor.render_pdf_bytes(bytes) {
        Ok(imgs) => imgs,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "Failed to render PDF");
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::to_value(MultiPageOcrResponse::error(
                    format!("Failed to render PDF: {}", e)
                )).unwrap()),
            );
        }
    };

    let render_time = start.elapsed();
    info!(
        request_id = %request_id,
        pages = images.len(),
        render_ms = render_time.as_secs_f64() * 1000.0,
        "PDF rendered to images"
    );

    // Process OCR on all pages
    let ocr_start = Instant::now();
    let results = match engine.process_multiple(images) {
        Ok(r) => r,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "OCR processing failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::to_value(MultiPageOcrResponse::error(
                    format!("OCR processing failed: {}", e)
                )).unwrap()),
            );
        }
    };

    let processing_time = ocr_start.elapsed();
    let total_time = start.elapsed();

    let total_regions: usize = results.iter().map(|r| r.text_regions.len()).sum();

    info!(
        request_id = %request_id,
        pages = results.len(),
        total_regions = total_regions,
        ocr_ms = processing_time.as_secs_f64() * 1000.0,
        total_ms = total_time.as_secs_f64() * 1000.0,
        "PDF OCR completed"
    );

    let response = OcrEngine::results_to_multipage_response(&results, processing_time.as_secs_f64() * 1000.0);

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap()))
}

/// Graceful shutdown signal handler
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C, starting graceful shutdown...");
        }
        _ = terminate => {
            info!("Received SIGTERM, starting graceful shutdown...");
        }
    }
}
