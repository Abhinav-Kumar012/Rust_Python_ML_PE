#![recursion_limit = "256"]
mod inference;
mod model;
mod state;

use axum::{
	routing::{get, post},
	Router,
};
use std::net::SocketAddr;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::inference::predict_handler;
use crate::state::AppState;

#[tokio::main]
async fn main() {
	// Initialize tracing
	tracing_subscriber::registry()
		.with(tracing_subscriber::EnvFilter::new(
			std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
		))
		.with(tracing_subscriber::fmt::layer())
		.init();

	// Load Model State
	let model_dir = std::env::var("MODEL_DIR")
		.unwrap_or_else(|_| -> String { "./model".to_string() });

	// Let the AppState construct the pre-loaded memory model
	let state = AppState::new(&model_dir);

	// Build Axum Router
	let app = Router::new()
		.route("/health", get(|| async { "OK" }))
		.route("/predict", post(predict_handler))
		.layer(CorsLayer::permissive())
		.layer(TraceLayer::new_for_http())
		.with_state(state);

	// Run Server
	let port = 9070;
	let addr = SocketAddr::from(([0, 0, 0, 0], port));
	tracing::info!("Server listening on http://{}", addr);

	let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
	axum::serve(listener, app).await.unwrap();
}
