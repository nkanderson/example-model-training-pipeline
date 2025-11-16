//////////////////////////////////////////////////////////////
// neural_network.sv - Multi-layer perceptron (MLP) for XOR
//
//
// Description:
// ------------
// This module implements a 2-layer neural network trained on XOR.
// Architecture: 2 inputs -> 2 hidden neurons (ReLU) -> 1 output neuron (linear)
// Uses Q3.12 fixed-point format for weights and computations.
//
//////////////////////////////////////////////////////////////

module neural_network #(parameter DATA_WIDTH = 16) (
    input logic [1:0] network_inputs,  // 2 1-bit inputs
    output logic signed [DATA_WIDTH-1:0] network_output  // 16-bit signed output in Q3.12 format
);

    // Hidden layer weights and biases (loaded from memory files)
    logic signed [DATA_WIDTH-1:0] hidden_weights_0 [3];  // Weights + bias for hidden neuron 0
    logic signed [DATA_WIDTH-1:0] hidden_bias_0;
    logic signed [DATA_WIDTH-1:0] hidden_weights_1 [3];  // Weights + bias for hidden neuron 1
    logic signed [DATA_WIDTH-1:0] hidden_bias_1;

    // Output layer weights and bias (loaded from memory files)
    logic signed [DATA_WIDTH-1:0] output_weights [3];  // Weights + bias for output neuron
    logic signed [DATA_WIDTH-1:0] output_bias;

    // Hidden layer outputs
    logic signed [DATA_WIDTH-1:0] hidden_outputs [2];
    
    // Unpacked arrays for connecting to hidden neurons
    logic hidden_inputs_0 [2];
    logic hidden_inputs_1 [2];
    
    // Assign individual bits to unpacked arrays
    assign hidden_inputs_0[0] = network_inputs[0];
    assign hidden_inputs_0[1] = network_inputs[1];
    assign hidden_inputs_1[0] = network_inputs[0];
    assign hidden_inputs_1[1] = network_inputs[1];

    // Load weights and biases from memory files
    initial begin
        $readmemh("weights/neuron_0.mem", hidden_weights_0);
        hidden_bias_0 = hidden_weights_0[2];  // Bias is last element
        
        $readmemh("weights/neuron_1.mem", hidden_weights_1);
        hidden_bias_1 = hidden_weights_1[2];  // Bias is last element
        
        $readmemh("weights/neuron_2.mem", output_weights);
        output_bias = output_weights[2];  // Bias is last element
    end

    // Instantiate hidden neurons
    hidden_neuron #(.NUM_INPUTS(2), .DATA_WIDTH(DATA_WIDTH)) hidden_0 (
        .inputs(hidden_inputs_0),
        .weights(hidden_weights_0[0:1]),  // First 2 elements are weights
        .bias(hidden_bias_0),
        .neuron_out(hidden_outputs[0])
    );

    hidden_neuron #(.NUM_INPUTS(2), .DATA_WIDTH(DATA_WIDTH)) hidden_1 (
        .inputs(hidden_inputs_1),
        .weights(hidden_weights_1[0:1]),  // First 2 elements are weights
        .bias(hidden_bias_1),
        .neuron_out(hidden_outputs[1])
    );

    // Instantiate output neuron
    output_neuron #(.NUM_INPUTS(2), .DATA_WIDTH(DATA_WIDTH)) output_0 (
        .inputs(hidden_outputs),
        .weights(output_weights[0:1]),  // First 2 elements are weights
        .bias(output_bias),
        .neuron_out(network_output)
    );

endmodule
