//////////////////////////////////////////////////////////////
// nn_tb.sv - Testbench for neural network XOR implementation
//
//
// Description:
// ------------
// Testbench for the multi-layer perceptron trained on XOR.
// Tests all 4 input combinations and verifies outputs.
// Output threshold: >= 0.5 maps to 1, < 0.5 maps to 0
//
//////////////////////////////////////////////////////////////

module nn_tb;
    // Parameters
    parameter DATA_WIDTH = 16;
    parameter FRACTIONAL_BITS = 12;
    
    // Testbench signals
    logic [1:0] inputs;
    logic signed [DATA_WIDTH-1:0] output_fixed;
    real output_float;
    logic output_binary;
    
    // Expected outputs for XOR
    logic expected_outputs [4] = '{0, 1, 1, 0};
    
    // Instantiate the neural network
    neural_network #(.DATA_WIDTH(DATA_WIDTH)) dut (
        .network_inputs(inputs),
        .network_output(output_fixed)
    );
    
    // Convert fixed-point to floating-point
    always_comb begin
        output_float = $itor(output_fixed) / (2.0 ** FRACTIONAL_BITS);
        // Threshold at 0.5
        output_binary = (output_fixed >= 16'sd2048) ? 1'b1 : 1'b0;
    end
    
    // Test stimulus
    initial begin
        // Generate VCD file for GTKWave
        $dumpfile("nn_tb.vcd");
        $dumpvars(0, nn_tb);
        
        $display("====================================");
        $display("Neural Network XOR Testbench");
        $display("Format: Q3.12 (16-bit signed)");
        $display("Threshold: 0.5 (2048 in fixed-point)");
        $display("====================================\n");
        
        // Test all 4 XOR cases
        for (int i = 0; i < 4; i++) begin
            inputs = i[1:0];
            #10; // Wait for combinational logic to settle
            
            $display("Test %0d: inputs = [%0d, %0d]", i, inputs[1], inputs[0]);
            $display("  Output (fixed): %0d (0x%04h)", output_fixed, output_fixed);
            $display("  Output (float): %0.6f", output_float);
            $display("  Output (binary): %0d", output_binary);
            $display("  Expected: %0d", expected_outputs[i]);
            
            // Check if output matches expected
            if (output_binary == expected_outputs[i]) begin
                $display("  PASS\n");
            end else begin
                $display("  FAIL\n");
            end
        end
        
        $display("====================================");
        $display("Test Complete");
        $display("====================================");
        $finish;
    end
    
endmodule
