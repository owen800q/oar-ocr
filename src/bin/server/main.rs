//! OAR-OCR Server and CLI
//!
//! A cross-platform binary for OCR processing via CLI or HTTP server.
//!
//! # Usage
//!
//! ## CLI Mode
//! ```bash
//! oar-ocr-server ocr --url "https://example.com/image.jpg" --det-model models/det.onnx --rec-model models/rec.onnx --dict-path models/dict.txt
//! oar-ocr-server ocr --file image.jpg --det-model models/det.onnx --rec-model models/rec.onnx --dict-path models/dict.txt
//! ```
//!
//! ## Server Mode
//! ```bash
//! oar-ocr-server serve --det-model models/det.onnx --rec-model models/rec.onnx --dict-path models/dict.txt --port 8080
//! ```

mod cli;
mod config;
mod ocr;
mod pdf;
mod server;

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "oar-ocr-server")]
#[command(author = "OAR-OCR Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "OCR processing via CLI or HTTP server", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Process a single image via CLI
    Ocr {
        /// URL of the image to process
        #[arg(long, conflicts_with = "file")]
        url: Option<String>,

        /// Local file path of the image to process
        #[arg(long, conflicts_with = "url")]
        file: Option<PathBuf>,

        /// Path to the text detection model
        #[arg(long = "det-model", env = "OAR_DET_MODEL")]
        det_model: PathBuf,

        /// Path to the text recognition model
        #[arg(long = "rec-model", env = "OAR_REC_MODEL")]
        rec_model: PathBuf,

        /// Path to the character dictionary
        #[arg(long = "dict-path", env = "OAR_DICT_PATH")]
        dict_path: PathBuf,

        /// Output format (json, text, pretty)
        #[arg(long, default_value = "pretty")]
        output: String,

        /// Device to use (cpu, cuda, cuda:0, etc.)
        #[arg(long, default_value = "cpu", env = "OAR_DEVICE")]
        device: String,
    },
    /// Start the HTTP server
    Serve {
        /// Path to the text detection model
        #[arg(long = "det-model", env = "OAR_DET_MODEL")]
        det_model: PathBuf,

        /// Path to the text recognition model
        #[arg(long = "rec-model", env = "OAR_REC_MODEL")]
        rec_model: PathBuf,

        /// Path to the character dictionary
        #[arg(long = "dict-path", env = "OAR_DICT_PATH")]
        dict_path: PathBuf,

        /// Port to listen on
        #[arg(long, short, default_value = "8080", env = "OAR_PORT")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "0.0.0.0", env = "OAR_HOST")]
        host: String,

        /// Device to use (cpu, cuda, cuda:0, etc.)
        #[arg(long, default_value = "cpu", env = "OAR_DEVICE")]
        device: String,

        /// Number of worker threads (defaults to number of CPUs)
        #[arg(long, env = "OAR_WORKERS")]
        workers: Option<usize>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize tracing
    oar_ocr::utils::init_tracing();

    let cli = Cli::parse();

    match cli.command {
        Commands::Ocr {
            url,
            file,
            det_model,
            rec_model,
            dict_path,
            output,
            device,
        } => {
            let config = config::OcrConfig {
                det_model,
                rec_model,
                dict_path,
                device,
            };

            if let Some(url) = url {
                info!("Processing URL: {}", url);
                cli::process_url(&url, &config, &output).await?;
            } else if let Some(file) = file {
                info!("Processing file: {}", file.display());
                cli::process_file(&file, &config, &output)?;
            } else {
                eprintln!("Error: Either --url or --file must be provided");
                std::process::exit(1);
            }
        }
        Commands::Serve {
            det_model,
            rec_model,
            dict_path,
            port,
            host,
            device,
            workers,
        } => {
            let config = config::ServerConfig {
                ocr: config::OcrConfig {
                    det_model,
                    rec_model,
                    dict_path,
                    device,
                },
                host,
                port,
                workers,
            };

            info!("Starting server on {}:{}", config.host, config.port);
            server::run_server(config).await?;
        }
    }

    Ok(())
}
