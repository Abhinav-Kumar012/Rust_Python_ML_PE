mod inference;
mod model;
mod state;

use axum::{
	Router,
	routing::{get, post},
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
	// Path is relative to where the binary is run. In Docker, it will be at root.
	// In local dev, likely ../../model/mnist_rust/model.mpk
	// We'll trust the user/env to run it from correct place or Docker.
	let model_path = std::env::var("MODEL_PATH")
		.unwrap_or_else(|_| -> String { "./model/mnist_rust/model.mpk".to_string() });
	dbg!(&model_path);

	let state = AppState::new(&model_path);

	// Build Router
	let app = Router::new()
		.route("/health", get(|| async { "OK" }))
		.route("/predict", post(predict_handler))
		.layer(CorsLayer::permissive())
		.layer(TraceLayer::new_for_http())
		.with_state(state);

	// Run Server
	let addr = SocketAddr::from(([0, 0, 0, 0], 9050));
	tracing::info!("listening on {}", addr);

	let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
	axum::serve(listener, app).await.unwrap();
}
