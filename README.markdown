# TensorFlow Lite native module for Veracruz

This is a TensorFlow Lite [native module for Veracruz](https://github.com/veracruz-project/veracruz/discussions/577).
It is meant to be executed by the [native module sandboxer](https://github.com/veracruz-project/native-module-sandboxer) in a sandbox environment everytime a WebAssembly program invokes it.  
Just like any native module, it is an entry point to a more complex library and only exposes preselected high-level features to the programs invoking it.

This native module takes an execution configuration file specifying:
* the input tensor's path (only one supported yet)
* the model's path
* the output tensor's path (only one supported yet)
* the number of CPU threads to use

It then performs inference on the model with the given input tensor and outputs a tensor.
Training is not supported yet.
