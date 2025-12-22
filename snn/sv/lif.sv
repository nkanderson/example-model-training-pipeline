// Leaky Integrate-and-Fire (LIF) Neuron Module
// Implements a spiking neuron with membrane potential integration, leak, and reset
// Uses multiplicative decay matching simplified snnTorch without current scaling:
// mem = beta * mem + current

module lif #(
    parameter THRESHOLD = 200,           // Spike threshold
    parameter BETA = 64,                 // Decay factor in Q1.7 format (64 = 0.5)
    parameter REFRACTORY_PERIOD = 0      // Clock cycles neuron is silent after spike
) (
    input wire clk,
    input wire reset,
    input wire enable,                   // Enable signal - only update when high
    input wire signed [7:0] current,     // Input current
    output reg spike                     // Output spike
);

    // Internal state
    reg signed [15:0] membrane_potential;  // Wider than current to prevent overflow during integration
    reg [3:0] refractory_counter;          // Counter for refractory period
    
    // Next state computation
    logic signed [15:0] next_membrane;
    logic signed [15:0] current_extended;
    logic signed [15:0] decay_potential;
    logic signed [23:0] decay_temp;

    // Combinational logic to compute next membrane potential
    // Implements: mem = beta * mem + current (matching simplied snnTorch without current scaling)
    always_comb begin
        // Sign-extend current from 8-bit to 16-bit
        current_extended = {{8{current[7]}}, current};
        
        // Step 1: Apply decay to current membrane (multiply by beta)
        // BETA is in Q1.7 format (8 bits: 1 integer bit, 7 fractional bits)
        // membrane_potential is 16-bit signed
        // Result of multiplication is 24-bit, need to shift right by 7 to get back to 16-bit
        decay_temp = membrane_potential * $signed({1'b0, BETA});
        decay_potential = decay_temp >>> 7;  // Arithmetic right shift to scale back
        
        // Step 2: Add input current
        next_membrane = decay_potential + current_extended;
    end

    // Sequential logic - update state on clock edge
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= 16'sd0;
            refractory_counter <= 4'd0;
            spike <= 1'b0;
        end else if (enable) begin
            spike <= 1'b0;  // Default: no spike
            
            if (refractory_counter > 0) begin
                // In refractory period - neuron is silent
                refractory_counter <= refractory_counter - 1;
                membrane_potential <= 16'sd0;  // Hold at zero during refractory
            end else begin
                // Check for spike threshold BEFORE updating membrane
                if (next_membrane >= THRESHOLD) begin
                    spike <= 1'b1;
                    membrane_potential <= membrane_potential - THRESHOLD;  // Reset after spike
                    refractory_counter <= REFRACTORY_PERIOD;
                end else begin
                    // Update membrane potential with computed next value
                    membrane_potential <= next_membrane;
                end
            end
        end
        // When enable=0, hold all state (membrane_potential, refractory_counter, spike unchanged)
    end

endmodule
