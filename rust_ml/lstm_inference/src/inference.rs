use axum::{
	extract::State,
	http::StatusCode,
	Json,
};
use burn::data::dataloader::batcher::Batcher;
use lstm_train::dataset::{SequenceBatcher, SequenceDatasetItem};
use serde::Serialize;

use crate::state::{AppState, Backend};

#[derive(Serialize)]
pub struct PredictResponse {
	pub predicted_next_value: f32,
}

#[derive(Serialize)]
pub struct ErrorResponse {
	pub error: String,
}

pub async fn predict_handler(
	State(state): State<AppState>,
	Json(payload): Json<SequenceDatasetItem>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
	let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();

	// Create batcher mapped to backend
	let batcher = SequenceBatcher::default();

	// Process item into batched tensors
	let batch = batcher.batch(vec![payload], &device);

	// Perform forward pass inference
	let output = state.model.lock().unwrap().forward(batch.sequences, None);

	// Extract single result
	let predicted_tensors = output.squeeze_dim::<1>(1).into_data();
	
	let predicted_value = predicted_tensors
		.as_slice::<f32>()
		.unwrap_or(&[])
		.first()
		.copied()
		.unwrap_or(0.0);

	Ok(Json(PredictResponse {
		predicted_next_value: predicted_value,
	}))
}
