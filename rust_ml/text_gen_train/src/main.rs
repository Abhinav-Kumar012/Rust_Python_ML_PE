#![recursion_limit = "256"]
mod data;
mod model;
mod training;
use crate::data::DbPediaDataset;
use burn::optim::decay::WeightDecayConfig;
use crate::{training::ExperimentConfig};
type Elem = f32;

type MyBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu<Elem,i32,u32>>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    crate::training::train::<MyBackend, DbPediaDataset>(
        MyBackend::default(),
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );
}