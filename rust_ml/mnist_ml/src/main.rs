mod data;
mod model;
mod training;

use crate::{model::ModelConfig, training::TrainingConfig};
use burn::{
	backend::{Autodiff, Wgpu},
	optim::AdamConfig,
};

fn main() {
	type MyBackend = Wgpu<f32, i32>;
	type MyAutodiffBackend = Autodiff<MyBackend>;

	let device = burn::backend::wgpu::WgpuDevice::default();
	let artifact_dir = "../model/mnist_rust";
	crate::training::train::<MyAutodiffBackend>(
		artifact_dir,
		TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
		device.clone(),
	);
}
