use axum::{
	Json,
	extract::{Multipart, State},
	http::StatusCode,
};
use burn::tensor::{Shape, Tensor, TensorData};
use image::ImageReader;
use serde::Serialize;
use std::io::Cursor;

use crate::state::{AppState, Backend};

#[derive(Serialize)]
pub struct InferenceResponse {
	prediction: usize,
}

pub async fn predict_handler(
	State(state): State<AppState>,
	mut multipart: Multipart,
) -> Result<Json<InferenceResponse>, (StatusCode, String)> {
	// 1. Extract image from multipart
	let field = multipart
		.next_field()
		.await
		.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?
		.ok_or((StatusCode::BAD_REQUEST, "No file uploaded".to_string()))?;

	let data = field
		.bytes()
		.await
		.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

	// 2. Load and preprocess image
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

	// Convert to tensor [1, 28, 28]
	let mut pixels = Vec::new();
	for pixel in img.to_luma8().pixels() {
		let val = pixel.0[0] as f32 / 255.0; // Normalize 0-1
		pixels.push(val);
	}

	// Create tensor: [Batch=1, Height=28, Width=28]
	// Use TensorData for explicit shape and avoid ambiguity
	let shape = Shape::new([1, 28, 28]);
	let data = TensorData::new(pixels, shape);
	let input = Tensor::<Backend, 3>::from_data(data, &Default::default());

	// 3. Run Inference
	// Acquire lock. Since this is std::sync::Mutex, we must not hold it across await points.
	// forward() is sync, so this is safe.
	let output = {
		let model = state
			.model
			.lock()
			.map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
		model.forward(input)
	};

	let prediction = output.argmax(1).into_scalar() as usize;

	Ok(Json(InferenceResponse { prediction }))
}
