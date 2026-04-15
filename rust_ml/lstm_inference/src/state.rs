use burn::{
	backend::Wgpu,
	module::Module,
	prelude::Config,
	record::{CompactRecorder, Recorder},
};
use crate::model::{LstmNetwork, LstmNetworkConfig};
use std::sync::{Arc, Mutex};

pub type MyBackend = Wgpu;

#[derive(Config,Debug)]
pub struct InferenceConfig {
	pub model: LstmNetworkConfig,
}

#[derive(Clone)]
pub struct AppState {
	pub model: Arc<Mutex<LstmNetwork<MyBackend>>>,
}

impl AppState {
	pub fn new(model_dir: &str) -> Self {
		let device = Default::default();

		let config_path = format!("{}/config.json", model_dir);
		let model_path = format!("{}/model", model_dir);

		// Load training configuration
		let config = InferenceConfig::load(&config_path)
			.expect("Config should exist for the model; run train first");

		// Load model configuration and initialized layers
		let record = CompactRecorder::new()
			.load(model_path.into(), &device)
			.expect("Trained model should exist; run train first");

		let model: LstmNetwork<MyBackend> = config.model.init(&device).load_record(record);

		Self {
			model: Arc::new(Mutex::new(model)),
		}
	}
}
