use burn::{
	backend::NdArray,
	module::Module,
	record::{NoStdTrainingRecorder, Recorder},
};
use regression::model::{RegressionModel, RegressionModelConfig, RegressionModelRecord};
use std::sync::{Arc, Mutex};

pub type Backend = NdArray;

#[derive(Clone)]
pub struct AppState {
	pub model: Arc<Mutex<RegressionModel<Backend>>>,
}

impl AppState {
	pub fn new(model_path: &str) -> Self {
		let device = Default::default();

		// Load model configuration
		let record: RegressionModelRecord<Backend> = NoStdTrainingRecorder::new()
			.load(model_path.into(), &device)
			.expect("Failed to load model record. Ensure the model is trained.");

		let model = RegressionModelConfig::new()
			.init(&device)
			.load_record(record);

		Self {
			model: Arc::new(Mutex::new(model)),
		}
	}
}
