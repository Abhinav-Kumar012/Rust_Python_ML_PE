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
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::data::{AgNewsDataset, TextClassificationBatcher};
use crate::model::TextClassificationModel;

// Define the concrete backend type we want to use (WGPU)
type MyBackend = Wgpu<f32, i32>;

// --- App State ---
struct AppState {
	// Wrap model in Mutex because Burn WGPU backend tensors might not be Sync (OnceCell)
	model: Arc<Mutex<TextClassificationModel<MyBackend>>>,
	batcher: Arc<TextClassificationBatcher>,
	device: WgpuDevice,
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
	let device = WgpuDevice::default();

	// Start server
	run_server(device).await;
}

async fn run_server(device: WgpuDevice) {
	let art_dir = std::env::var("ARTIFACT_DIR")
		.unwrap_or_else(|_| "./model/text_classification_ag_news_rust".to_string());

	println!("Loading model from: {}", art_dir);

	// Load model
	let model = crate::inference::load_model::<MyBackend, AgNewsDataset>(&art_dir, device.clone());
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
async fn handle_predict(
	State(state): State<Arc<AppState>>,
	Json(payload): Json<PredictRequest>,
) -> impl IntoResponse {
	// Lock the model execution
	let model = state.model.lock().await;

	let (text, prediction) = crate::inference::infer_one::<MyBackend, AgNewsDataset>(
		&model,
		state.batcher.clone(),
		state.device.clone(),
		payload.text,
	);

	Json(PredictResponse { text, prediction })
}
