use axum::{
	Json,
	extract::{Multipart, State},
	http::StatusCode,
};
use burn::tensor::{Shape, Tensor, TensorData};
use chrono::Utc;
use image::ImageReader;
use serde::Serialize;
use std::io::Cursor;
use std::time::Instant;
use sysinfo::System;

use crate::state::{AppState, Backend};

#[derive(Serialize)]
pub struct InferenceResponse {
	prediction: usize,
	meta: RequestMetrics,
}

#[derive(Serialize)]
struct RequestMetrics {
	timestamp: String,
	latency_ms: f64,
	resources: ResourceMetrics,
	security_context: SecurityContext,
}

#[derive(Serialize)]
struct ResourceMetrics {
	global_cpu_usage_percent: f32,
	available_memory_mb: u64,
}

#[derive(Serialize)]
struct SecurityContext {
	service_version: String,
	model_version: String,
}

pub async fn predict_handler(
	State(state): State<AppState>,
	mut multipart: Multipart,
) -> Result<Json<InferenceResponse>, (StatusCode, String)> {
	let start_time = Instant::now();

	// 1. Capture Resources (Snapshot)
	// Ref: sysinfo 0.30+ changes
	let mut sys = System::new_all();
	sys.refresh_all();

	let cpu_usage = sys.global_cpu_info().cpu_usage();
	let mem_available = sys.available_memory() / 1024 / 1024; // MB

	// 2. Extract image from multipart
	let field = multipart
		.next_field()
		.await
		.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
		.ok_or((StatusCode::BAD_REQUEST, "No file uploaded".to_string()))?;

	let data = field
		.bytes()
		.await
		.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

	// 3. Load and preprocess image
	let img = ImageReader::new(Cursor::new(data))
		.with_guessed_format()
		.map_err(|e| {
			(
				StatusCode::BAD_REQUEST,
				format!("Failed to read image format: {}", e),
			)
		})?
		.decode()
		.map_err(|e| {
			(
				StatusCode::BAD_REQUEST,
				format!("Failed to decode image: {}", e),
			)
		})?;

	// Convert to grayscale and resize to 28x28
	let img = img.grayscale();
	let img = img.resize_exact(28, 28, image::imageops::FilterType::Lanczos3);

	// Convert to tensor [1, 28, 28] using TensorData
	let mut pixels = Vec::new();
	for pixel in img.to_luma8().pixels() {
		let val = pixel.0[0] as f32 / 255.0;
		let normalized = (val - 0.1307) / 0.3081;
		pixels.push(normalized);
	}

	let shape = Shape::new([1, 28, 28]);
	let data = TensorData::new(pixels, shape);
	let input = Tensor::<Backend, 3>::from_data(data, &Default::default());

	// 4. Run Inference
	let output = {
		let model = state
			.model
			.lock()
			.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
		model.forward(input)
	};

	let prediction = output.argmax(1).into_scalar() as usize;

	// 5. Calculate Metrics
	let duration = start_time.elapsed();
	let latency_ms = duration.as_secs_f64() * 1000.0;

	let metrics = RequestMetrics {
		timestamp: Utc::now().to_rfc3339(),
		latency_ms,
		resources: ResourceMetrics {
			global_cpu_usage_percent: cpu_usage,
			available_memory_mb: mem_available,
		},
		security_context: SecurityContext {
			service_version: env!("CARGO_PKG_VERSION").to_string(),
			model_version: "v1-sha256-placeholder".to_string(), // Ideally injected or computed hash of model file
		},
	};

	// Log to stdout (JSON structured log)
	let log_entry = serde_json::json!({
		"event": "inference_complete",
		"prediction": prediction,
		"metrics": metrics
	});
	println!("{}", log_entry);

	Ok(Json(InferenceResponse {
		prediction,
		meta: metrics,
	}))
}
