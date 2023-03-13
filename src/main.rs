//! A native module for ML inference on TensorFlow Lite.
//! Takes an input tensor, feeds it to the model and outputs an output tensor.
//!
//! ## Authors
//!
//! The Veracruz Development Team.
//!
//! ## Licensing and copyright notice
//!
//! See the `LICENSE_MIT.markdown` file in the Veracruz root directory for
//! information on licensing and copyright.

use anyhow;
use libc::c_int;
use serde::Deserialize;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tflite::ops::builtin::BuiltinOpResolver;
use tflite::{FlatBufferModel, InterpreterBuilder};
use std::time::{SystemTime, UNIX_EPOCH};

/// Module's API.
#[derive(Deserialize, Debug)]
pub(crate) struct TfLiteInferenceService {
    // TODO: support several inputs and outputs
    /// Path to the input tensor to be fed to the network.
    input_tensor_path: PathBuf,
    /// Path to the model serialized with FlatBuffers.
    model_path: PathBuf,
    /// Path to the output tensor containing the result of the prediction.
    output_tensor_path: PathBuf,
    /// Number of CPU threads to use for the TensorFlow Lite interpreter.
    num_threads: c_int,
}

impl TfLiteInferenceService {
    /// Create a new service, with empty internal state.
    pub fn new() -> Self {
        Self {
            input_tensor_path: PathBuf::new(),
            model_path: PathBuf::new(),
            output_tensor_path: PathBuf::new(),
            num_threads: -1,
        }
    }

    /// Try to parse input into a TfLiteInferenceService structure.
    /// An attacker may inject malformed paths but that should be caught by the
    /// VFS when attempting to access the corresponding files.
    /// The input tensor might not match the model's input dimensions, but this
    /// will be caught by TensorFlow Lite.
    fn try_parse(&mut self, input: &[u8]) -> anyhow::Result<bool> {
        let deserialized_input: TfLiteInferenceService =
            match postcard::from_bytes(&input) {
                Ok(o) => o,
                Err(_) => return Ok(false),
            };
        *self = deserialized_input;
        Ok(true)
    }

    /// The core service. It loads the model pointed by `model_path` then feeds
    /// the input read from `input_tensor_path` to the model, and writes the
    /// resulting tensor to the file at `output_tensor_path`.
    /// The interpreter can be further configured with `num_threads`.
    fn infer(&mut self) -> anyhow::Result<()> {
        let TfLiteInferenceService {
            input_tensor_path,
            model_path,
            output_tensor_path,
            num_threads,
        } = self;

        // Build model and interpreter
        let model = FlatBufferModel::build_from_file(model_path)?;
        let resolver = BuiltinOpResolver::default();
        let builder = InterpreterBuilder::new(&model, &resolver)?;
        let mut interpreter = builder.build()?;

        // Configure interpreter
        interpreter.set_num_threads(*num_threads);

        interpreter.allocate_tensors()?;

        // Load and configure inputs.
        // XXX: We assume a single input for now
        let inputs = interpreter.inputs().to_vec();
        assert_eq!(inputs.len(), 1);
        let input_index = inputs[0];
        let mut input_file = File::open(&input_tensor_path)?;
        input_file.read_exact(
            interpreter.tensor_data_mut(input_index)?
        )?;

        println!("invoking...");
        interpreter.invoke()?;

        // Get outputs
        // XXX: We assume a single output for now
        let outputs = interpreter.outputs().to_vec();
        let output_index = outputs[0];
        let output = interpreter
            .tensor_data(output_index)?;

        println!("writing results...");
        let mut file = File::create(Path::new("/").join(output_tensor_path))?;
        file.write_all(&output.to_vec())?;

        Ok(())
    }
}

fn main() -> anyhow::Result<()>
{
    let mut service = TfLiteInferenceService::new();

    // Read input from execution configuration file
    println!("opening execution configuration file...");
    let mut f = File::open("/execution_config")?;
    let mut input = Vec::new();
    println!("reading execution configuration file...");
    f.read_to_end(&mut input)?;
    println!("parsing input...");
    service.try_parse(&input)?;
    service.infer()
}
