#![recursion_limit = "256"]
mod data;
mod inference;
mod model;
mod training;
type ElemType = f32;
use crate::data::AgNewsDataset;
use burn::tensor::backend::Backend;
mod wgpu {
	use crate::{ElemType, launch};
	use burn::backend::wgpu::{Wgpu, WgpuDevice};

	pub fn run() {
		launch::<Wgpu<ElemType, i32>>(WgpuDevice::default());
	}
}
pub fn launch<B: Backend>(device: B::Device) {
	// crate::inference::infer::<B, AgNewsDataset>(
	//     device,
	//     "/home/abhinav/Downloads/Rust_Python_ML_PE/rust_ml/model/text_classification_ag_news_rust",
	//     // Samples from the test dataset, but you are free to test with your own text.
	//     vec![
	//         "Jays power up to take finale Contrary to popular belief, the power never really \
	//          snapped back at SkyDome on Sunday. The lights came on after an hour delay, but it \
	//          took some extra time for the batting orders to provide some extra wattage."
	//             .to_string(),
	//         "Yemen Sentences 15 Militants on Terror Charges A court in Yemen has sentenced one \
	//          man to death and 14 others to prison terms for a series of attacks and terrorist \
	//          plots in 2002, including the bombing of a French oil tanker."
	//             .to_string(),
	//         "IBM puts grids to work at U.S. Open IBM will put a collection of its On \
	//          Demand-related products and technologies to this test next week at the U.S. Open \
	//          tennis championships, implementing a grid-based infrastructure capable of running \
	//          multiple workloads including two not associated with the tournament."
	//             .to_string(),
	//     ],
	// );
}
fn main() {
	wgpu::run();
}
