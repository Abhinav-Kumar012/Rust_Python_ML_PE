use std::sync::Arc;

use crate::{
	ModelConfig,
	data::{MnistBatch, MnistBatcher},
	model::Model,
	training,
};
use burn::{
	data::{
		dataloader::DataLoaderBuilder,
		dataset::{
			Dataset, InMemDataset,
			transform::{PartialDataset, SelectionDataset},
			vision::MnistDataset,
		},
	},
	nn::loss::CrossEntropyLossConfig,
	optim::AdamConfig,
	prelude::*,
	record::CompactRecorder,
	tensor::backend::AutodiffBackend,
	train::{
		ClassificationOutput, LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, ValidStep,
		metric::{
			AccuracyMetric, ClassReduction, CpuMemory, CpuTemperature, CpuUse, FBetaScoreMetric,
			LossMetric, PrecisionMetric, RecallMetric, TopKAccuracyMetric,
		},
	},
};
impl<B: Backend> Model<B> {
	pub fn forward_classification(
		&self,
		images: Tensor<B, 3>,
		targets: Tensor<B, 1, Int>,
	) -> ClassificationOutput<B> {
		let output = self.forward(images);
		let loss = CrossEntropyLossConfig::new()
			.init(&output.device())
			.forward(output.clone(), targets.clone());

		ClassificationOutput::new(loss, output, targets)
	}
}
impl<B: AutodiffBackend> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
	fn step(
		&self,
		batch: MnistBatch<B>,
	) -> TrainOutput<ClassificationOutput<B>> {
		let item = self.forward_classification(batch.images, batch.targets);

		TrainOutput::new(self, item.loss.backward(), item)
	}
}

impl<B: Backend> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B> {
	fn step(
		&self,
		batch: MnistBatch<B>,
	) -> ClassificationOutput<B> {
		self.forward_classification(batch.images, batch.targets)
	}
}

#[derive(Config, Debug)]
pub struct TrainingConfig {
	pub model: ModelConfig,
	pub optimizer: AdamConfig,
	#[config(default = 10)]
	pub num_epochs: usize,
	#[config(default = 64)]
	pub batch_size: usize,
	#[config(default = 4)]
	pub num_workers: usize,
	#[config(default = 42)]
	pub seed: u64,
	#[config(default = 1.0e-4)]
	pub learning_rate: f64,
	#[config(default = 0.8)]
	pub split: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
	// Remove existing artifacts before to get an accurate learner summary
	std::fs::remove_dir_all(artifact_dir).ok();
	std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
	artifact_dir: &str,
	config: TrainingConfig,
	device: B::Device,
) {
	create_artifact_dir(artifact_dir);
	config
		.save(format!("{artifact_dir}/config.json"))
		.expect("Config should be saved successfully");

	B::seed(&device, config.seed);

	let batcher = MnistBatcher::default();
	let original_train_dataset = Arc::new(MnistDataset::train());
	let original_test_dataset = Arc::new(MnistDataset::test());

	// For demonstration, let's assume you want to split the *training* dataset further into
	// an 80:20 train/validation set, or combine and then split.
	// If you only want to use the pre-defined train and test splits, you would just use
	// original_train_dataset and original_test_dataset directly.

	let total_len = original_train_dataset.len();

	// Calculate split points for 80:20 from the original training set
	let train_len = (total_len as f64 * config.split) as usize;
	let valid_len = total_len - train_len;

	let custom_train_split = PartialDataset::new(original_train_dataset.clone(), 0, train_len);
	let custom_valid_split =
		PartialDataset::new(original_train_dataset.clone(), train_len, total_len);

	let dataloader_train = DataLoaderBuilder::new(batcher.clone())
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(custom_train_split);

	let dataloader_test = DataLoaderBuilder::new(batcher)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(custom_valid_split);
	let f1metric = FBetaScoreMetric::multiclass(1.0, 1, ClassReduction::Macro);
	let precissionmetric = PrecisionMetric::multiclass(1, ClassReduction::Macro);
	let recallmetric = RecallMetric::multiclass(1, ClassReduction::Macro);

	let learner = LearnerBuilder::new(artifact_dir)
		.metric_train_numeric(AccuracyMetric::new())
		.metric_valid_numeric(AccuracyMetric::new())
		.metric_train_numeric(LossMetric::new())
		.metric_valid_numeric(LossMetric::new())
		.metric_train_numeric(precissionmetric.clone())
		.metric_valid_numeric(precissionmetric.clone())
		.metric_train_numeric(recallmetric.clone())
		.metric_valid_numeric(recallmetric.clone())
		.metric_train_numeric(f1metric.clone()) // F1 score
		.metric_valid_numeric(f1metric.clone())
		.metric_train_numeric(TopKAccuracyMetric::new(5))
		.metric_valid_numeric(TopKAccuracyMetric::new(5))
		// system / hardware
		.metric_train_numeric(CpuUse::new())
		.metric_valid_numeric(CpuUse::new())
		.metric_train_numeric(CpuTemperature::new())
		.metric_valid_numeric(CpuTemperature::new())
		.metric_train_numeric(CpuMemory::new())
		.metric_valid_numeric(CpuMemory::new())
		.with_file_checkpointer(CompactRecorder::new())
		.learning_strategy(LearningStrategy::SingleDevice(device.clone()))
		.num_epochs(config.num_epochs)
		.summary()
		.build(
			config.model.init::<B>(&device),
			config.optimizer.init(),
			config.learning_rate,
		);

	let result = learner.fit(dataloader_train, dataloader_test);

	result
		.model
		.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
		.expect("Trained model should be saved successfully");
}
