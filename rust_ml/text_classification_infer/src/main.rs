#![recursion_limit = "256"]
mod data;
mod inference;
mod model;
mod training;

use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

use axum::{
	Router,
	extract::{Json, State},
	response::IntoResponse,
	routing::post,
};
// use burn::backend::Cuda;
use burn::backend::{NdArray};//, wgpu::Wgpu};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
// use burn::tensor::backend::Backend;

use crate::data::{AgNewsDataset, TextClassificationBatcher};
use crate::model::TextClassificationModel;

// Define the concrete backend type we want to use (WGPU)
// type MyBackend = Wgpu<f32, i32>;
type MyBackend = NdArray<f32, i32>;

// --- App State ---
struct AppState<B: Backend> {
	// Wrap model in Mutex because Burn WGPU backend tensors might not be Sync (OnceCell)
	model: Arc<Mutex<TextClassificationModel<B>>>,
	batcher: Arc<TextClassificationBatcher>,
	device: B::Device,
}

// --- Request/Response DTOs ---
#[derive(Debug, Deserialize)]
struct PredictRequest {
	text: String,
}

#[derive(Debug, Serialize)]
struct PredictResponse {
	text: String,
	prediction: String,
}

// --- Main ---
#[tokio::main]
async fn main() {
	// Initialize tracing
	tracing_subscriber::fmt::init();

	// Initialize WGPU Device
	let device = <MyBackend as Backend>::Device::default();

	// Start server
	run_server::<MyBackend>(device).await;
}

async fn run_server<B: Backend>(device: B::Device) {
	let art_dir = std::env::var("ARTIFACT_DIR")
		.unwrap_or_else(|_| "./model/text_classification_ag_news_rust".to_string());

	println!("Loading model from: {}", art_dir);

	// Load model
	let model = crate::inference::load_model::<B, AgNewsDataset>(&art_dir, device.clone());
	let batcher = crate::inference::make_batcher(&art_dir);

	let shared_state = Arc::new(AppState {
		model: Arc::new(Mutex::new(model)),
		batcher: Arc::new(batcher),
		device,
	});

	let app = Router::new()
		.route("/predict", post(handle_predict))
		.layer(CorsLayer::permissive())
		.layer(TraceLayer::new_for_http())
		.with_state(shared_state);

	let port = 9050;
	let addr = SocketAddr::from(([0, 0, 0, 0], port));
	println!("Listening on {}", addr);

	let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
	axum::serve(listener, app).await.unwrap();
}

// --- Handlers ---
async fn handle_predict<B: Backend>(
	State(state): State<Arc<AppState<B>>>,
	Json(payload): Json<PredictRequest>,
) -> impl IntoResponse {
	// Lock the model execution
	let model = state.model.lock().await;

	let (text, prediction) = crate::inference::infer_one::<B, AgNewsDataset>(
		&model,
		state.batcher.clone(),
		state.device.clone(),
		payload.text,
	);

	Json(PredictResponse { text, prediction })
}
