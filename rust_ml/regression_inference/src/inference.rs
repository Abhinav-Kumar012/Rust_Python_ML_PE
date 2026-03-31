use axum::{
	extract::State,
	http::StatusCode,
	Json,
};
use burn::data::dataloader::batcher::Batcher;
use regression::dataset::{HousingBatcher, HousingDistrictItem};
use serde::Serialize;

use crate::state::{AppState, Backend};

#[derive(Serialize)]
pub struct PredictResponse {
	pub predicted_median_house_value: f32,
}

#[derive(Serialize)]
pub struct ErrorResponse {
	pub error: String,
}

pub async fn predict_handler(
	State(state): State<AppState>,
	Json(payload): Json<HousingDistrictItem>,
) -> Result<Json<PredictResponse>, (StatusCode, Json<ErrorResponse>)> {
	let device: <Backend as burn::tensor::backend::Backend>::Device = Default::default();

	// Create batcher mapped to backend
	let batcher = HousingBatcher::<Backend>::new(device.clone());

	// Process item
	// Note: HousingBatcher::batch transforms a Vec<Item> into a HousingBatch
	let batch = batcher.batch(vec![payload], &device);

	// Perform forward pass inference
	let output = state.model.lock().unwrap().forward(batch.inputs);

	// Extract single result
	let predicted_tensors = output.squeeze_dim::<1>(1).into_data();
	
	// Assuming `into_data()` gives us Burn's generic `Data`, we extract f32 value.
	// Since we batched a single item, it should be the first entry
	let predicted_value = predicted_tensors
		.iter::<f32>()
		.next()
		.unwrap_or(0.0);

	Ok(Json(PredictResponse {
		predicted_median_house_value: predicted_value,
	}))
}
