#![recursion_limit = "256"]
mod data;
mod inference;
mod model;
mod training;
use data::AgNewsDataset;

use crate::training::ExperimentConfig;
use burn::backend::{Autodiff, Wgpu};
use burn::{
	nn::transformer::TransformerEncoderConfig,
	optim::{AdamConfig, decay::WeightDecayConfig},
	tensor::backend::AutodiffBackend,
};

type ElemType = f32;

pub fn launch<B: AutodiffBackend>(devices: Vec<B::Device>) {
	let config = ExperimentConfig::new(
		TransformerEncoderConfig::new(256, 1024, 8, 4)
			.with_norm_first(true)
			.with_quiet_softmax(true),
		AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
	);

	crate::training::train::<B, AgNewsDataset>(
		devices,
		AgNewsDataset::train(),
		AgNewsDataset::test(),
		config,
		"./model/text_classification_ag_news_rust",
	);
}

fn main() {
	launch::<Autodiff<Wgpu<ElemType, i32>>>(vec![Default::default()]);
}
