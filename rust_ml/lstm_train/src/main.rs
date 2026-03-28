#![recursion_limit = "256"]
use burn::{
	grad_clipping::GradientClippingConfig, optim::AdamConfig, tensor::backend::AutodiffBackend,
};
use lstm_train::{model::LstmNetworkConfig, training::TrainingConfig};

const ARTIFACT_DIR: &str = "model/lstm_train";
pub fn launch<B: AutodiffBackend>(device: B::Device) {
	let config = TrainingConfig::new(
		LstmNetworkConfig::new(),
		// Gradient clipping via optimizer config
		AdamConfig::new().with_grad_clipping(Some(GradientClippingConfig::Norm(1.0))),
	);

	lstm_train::training::train::<B>(ARTIFACT_DIR, config, device);
}
mod wgpu {
	use crate::launch;
	use burn::backend::{Autodiff, wgpu::Wgpu};

	pub fn run() {
		launch::<Autodiff<Wgpu>>(Default::default());
	}
}

fn main() {
	wgpu::run();
}
