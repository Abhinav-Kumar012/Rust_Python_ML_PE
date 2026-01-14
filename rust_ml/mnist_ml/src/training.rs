use crate::{
	ModelConfig,
	data::{MnistBatch, MnistBatcher},
	model::Model,
};
use burn::{
	data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
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
use std::env;
use std::fs;
use std::time::Instant;
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

	let dataloader_train = DataLoaderBuilder::new(batcher.clone())
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(MnistDataset::train());

	let dataloader_test = DataLoaderBuilder::new(batcher)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(MnistDataset::test());
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

	let start_time = Instant::now();

	let result = learner.fit(dataloader_train, dataloader_test);

	let duration = start_time.elapsed();
	let duration_secs = duration.as_secs_f64();
	let avg_epoch_duration = duration_secs / config.num_epochs as f64;

	result
		.model
		.save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
		.expect("Trained model should be saved successfully");

	// Collect metrics
	let os_info = env::consts::OS;
	let arch_info = env::consts::ARCH;

	let model_path = format!("{artifact_dir}/model.mpk");
	let mut model_size = 0;
	if let Ok(metadata) = fs::metadata(&model_path) {
		model_size = metadata.len();
	} else {
		// Try without extension or with different extension if recorder differs,
		// but CompactRecorder usually uses binary format.
		// burn-train usually saves as `model.mpk` (msgpack) or `model.json` depending on recorder.
		// CompactRecorder uses generic serializer. Let's check directory or just try generic path.
		// Actually save_file adds extension. Default for CompactRecorder is mpk (msgpack) usually?
		// Let's assume .mpk based on common Burn usage or check file existence.
		// For safety we can just list dir or try standard extensions.
		// Using a simple check.
		if let Ok(metadata) = fs::metadata(format!("{artifact_dir}/model.mpk")) {
			model_size = metadata.len();
		}
	}

	// Manual JSON construction
	let json_content = format!(
		r#"{{
    "timestamp": {},
    "status": "success",
    "total_duration_sec": {:.2},
    "avg_epoch_duration_sec": {:.2},
    "setup_info": {{
        "seed": {}
    }},
    "environment": {{
        "os": "{}",
        "arch": "{}"
    }},
    "artifact_metrics": {{
        "model_size_bytes": {}
    }}
}}"#,
		std::time::SystemTime::now()
			.duration_since(std::time::UNIX_EPOCH)
			.unwrap_or_default()
			.as_secs(),
		duration_secs,
		avg_epoch_duration,
		config.seed,
		os_info,
		arch_info,
		model_size
	);

	fs::write(format!("{artifact_dir}/metrics.json"), json_content)
		.expect("Metrics should be saved successfully");
}
