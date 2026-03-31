use burn::{
	backend::NdArray,
	module::Module,
	prelude::Config,
	record::{CompactRecorder, Recorder},
};
use lstm_train::{
	model::LstmNetwork,
	training::TrainingConfig,
};
use std::sync::{Arc, Mutex};

pub type Backend = NdArray;

#[derive(Clone)]
pub struct AppState {
	pub model: Arc<Mutex<LstmNetwork<Backend>>>,
}

impl AppState {
	pub fn new(model_dir: &str) -> Self {
		let device = Default::default();

		let config_path = format!("{}/config.json", model_dir);
		let model_path = format!("{}/model", model_dir);

		// Load training configuration
		let config = TrainingConfig::load(&config_path)
			.expect("Config should exist for the model; run train first");

		// Load model configuration and initialized layers
		let record = CompactRecorder::new()
			.load(model_path.into(), &device)
			.expect("Trained model should exist; run train first");

		let model: LstmNetwork<Backend> = config.model.init(&device).load_record(record);

		Self {
			model: Arc::new(Mutex::new(model)),
		}
	}
}
