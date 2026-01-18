use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::module::Module;
use burn::record::CompactRecorder;
use std::sync::{Arc, Mutex};

use crate::model::{Model, ModelConfig};

pub type Backend = NdArray<f32>;

#[derive(Clone)]
pub struct AppState {
	// Model might not be Sync or we want to avoid deep cloning.
	// Wrap in Arc for cheap clone.
	// Wrap in Mutex to ensure Sync if backend/layers are !Sync.
	pub model: Arc<Mutex<Model<Backend>>>,
}

impl AppState {
	pub fn new(model_path: &str) -> Self {
		let device = NdArrayDevice::Cpu;

		// Initialize a default model first
		let config = ModelConfig::new(10, 512); // Defaults from training
		let model = config.init(&device);

		// Load pre-trained weights
		// Using CompactRecorder as used in training
		let recorder = CompactRecorder::new();
		let model = model
			.load_file(model_path, &recorder, &device)
			.expect("Failed to load model weights");

		Self {
			model: Arc::new(Mutex::new(model)),
		}
	}
}
