#![recursion_limit = "256"]

use burn::{backend::Autodiff, tensor::backend::Backend};
use regression::{inference, training};

static ARTIFACT_DIR: &str = "model/regression_train";
mod wgpu {
	use burn::backend::wgpu::{Wgpu, WgpuDevice};

	pub fn run() {
		let device = WgpuDevice::default();
		super::run::<Wgpu>(device);
	}
}
/// Train a regression model and predict results on a number of samples.
pub fn run<B: Backend>(device: B::Device) {
	training::run::<Autodiff<B>>(ARTIFACT_DIR, device.clone());
	inference::infer::<B>(ARTIFACT_DIR, device)
}
fn main() {
	wgpu::run();
}
