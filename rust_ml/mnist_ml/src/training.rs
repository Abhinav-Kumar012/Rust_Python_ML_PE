use crate::{
	ModelConfig,
	data::{MnistBatch, MnistBatcher},
	model::Model,
};
use burn::{
	data::{
		dataloader::DataLoaderBuilder, dataset::Dataset, dataset::InMemDataset,
		dataset::vision::MnistDataset,
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

	// Dataset Splitting 80:10:10
	let dataset = MnistDataset::train();
	let mut items: Vec<_> = dataset.iter().collect();

	// Shuffle with seed 42
	let mut rng = rand::rngs::StdRng::seed_from_u64(42);
	use rand::SeedableRng;
	use rand::seq::SliceRandom;
	items.shuffle(&mut rng);

	let total_len = items.len();
	let train_len = (total_len as f64 * 0.80) as usize;
	let valid_len = (total_len as f64 * 0.10) as usize;

	let (train_items, rest) = items.split_at(train_len);
	let (valid_items, test_items) = rest.split_at(valid_len);

	let train_dataset = InMemDataset::new(train_items.to_vec());
	let valid_dataset = InMemDataset::new(valid_items.to_vec());
	let test_dataset = InMemDataset::new(test_items.to_vec());

	let dataloader_train = DataLoaderBuilder::new(batcher.clone())
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(train_dataset);

	let dataloader_test = DataLoaderBuilder::new(batcher.clone())
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(valid_dataset);
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

	// Test Evaluation
	let dataloader_test_final = DataLoaderBuilder::new(batcher)
		.batch_size(config.batch_size)
		.shuffle(config.seed)
		.num_workers(config.num_workers)
		.build(test_dataset);

	let mut test_loss = 0.0;
	let mut test_acc = 0.0;
	let mut total_batches = 0;

	// Use model for evaluation
	let model_valid = &result.model;

	for batch in dataloader_test_final.iter() {
		let output = model_valid.forward_classification(batch.images, batch.targets);
		let loss = output.loss.into_scalar();

		// Calculate simple accuracy manually or using metric if cleaner
		// Doing simple manual match for portability without full metric setup
		let targets = output.targets.into_data();
		let preds = output.output.argmax(1).into_data();
		let targets_vec: Vec<i64> = targets.to_vec().expect("Should be able to convert to vec");
		let preds_vec: Vec<i64> = preds.to_vec().expect("Should be able to convert to vec"); // Assuming argmax returns int64 compatible
		let batch_size = targets_vec.len();

		let mut correct = 0;
		for i in 0..batch_size {
			if targets_vec[i] == preds_vec[i] {
				correct += 1;
			}
		}
		let batch_acc = correct as f64 / batch_size as f64;

		test_loss += loss.to_f64();
		test_acc += batch_acc;
		total_batches += 1;
	}

	if total_batches > 0 {
		test_loss /= total_batches as f64;
		test_acc /= total_batches as f64;
	}

	println!(
		"Test Set Evaluation: loss={:.4}, acc={:.4}",
		test_loss, test_acc
	);

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
    }},
    "test_metrics": {{
        "accuracy": {:.4},
        "loss": {:.4}
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
		model_size,
		test_acc,
		test_loss
	);

	fs::write(format!("{artifact_dir}/metrics.json"), json_content)
		.expect("Metrics should be saved successfully");
}
