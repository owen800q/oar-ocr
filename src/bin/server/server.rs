//! HTTP server for OCR processing.

use crate::config::ServerConfig;
use crate::ocr::{download_image, OcrEngine, OcrRequest, OcrResponse, SharedOcrEngine};
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
    info!("  POST /ocr        - OCR processing");
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
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// OCR processing endpoint
async fn ocr_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OcrRequest>,
) -> impl IntoResponse {
    let request_id = uuid::Uuid::new_v4().to_string();
    info!(request_id = %request_id, url = %request.url, "Processing OCR request");

    let start = Instant::now();

    // Download image
    let image = match download_image(&request.url).await {
        Ok(img) => img,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "Failed to download image");
            return (
                StatusCode::BAD_REQUEST,
                Json(OcrResponse::error(format!("Failed to download image: {}", e))),
            );
        }
    };

    let download_time = start.elapsed();
    info!(
        request_id = %request_id,
        width = image.width(),
        height = image.height(),
        download_ms = download_time.as_secs_f64() * 1000.0,
        "Image downloaded"
    );

    // Process OCR
    let ocr_start = Instant::now();
    let result = match state.engine.process(image) {
        Ok(r) => r,
        Err(e) => {
            error!(request_id = %request_id, error = %e, "OCR processing failed");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(OcrResponse::error(format!("OCR processing failed: {}", e))),
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

    (StatusCode::OK, Json(response))
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
