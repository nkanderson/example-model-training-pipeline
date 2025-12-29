// Leaky Integrate-and-Fire (LIF) Neuron Module
// Implements a spiking neuron with membrane potential integration, leak, and reset
// Matches snnTorch Leaky neuron with reset_mechanism="subtract" and reset_delay=True
//
// Membrane dynamics: mem[t] = beta * mem[t-1] + input[t] - reset[t-1] * threshold
// Spike generation: spike[t] = (mem[t] >= threshold)
// Reset delay: The threshold subtraction is applied one cycle after the spike
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
// - Threshold 1.0 = 8192 (2^13)
// - Beta 0.9 in Q1.7 = 115

module lif #(
    parameter THRESHOLD = 8192,          // Spike threshold (1.0 in QS2.13)
    parameter BETA = 115                 // Decay factor in Q1.7 format (115 ≈ 0.9)
) (
    input wire clk,
    input wire reset,
    input wire enable,                   // Enable signal - only update when high
    input wire signed [15:0] current,    // Input current (QS2.13 format)
    output logic spike                     // Output spike
);

    // Internal state
    logic signed [23:0] membrane_potential;  // Membrane potential (wider for accumulation headroom)

    // Next state computation
    logic signed [23:0] next_membrane;
    logic signed [23:0] current_extended;
    logic signed [23:0] decay_potential;
    logic signed [31:0] decay_temp;
    logic signed [23:0] reset_subtract;

    // Combinational logic to compute next membrane potential
    // Implements: mem = beta * mem + current - prev_spike * threshold
    always_comb begin
        // Sign-extend current from 16-bit to 24-bit
        current_extended = {{8{current[15]}}, current};

        // Step 1: Apply decay to current membrane (multiply by beta)
        // BETA is in Q1.7 format (8 bits: 1 integer bit, 7 fractional bits)
        // membrane_potential is 24-bit signed (QS2.13 with extra headroom)
        // Result of multiplication is 32-bit, shift right by 7 to scale back
        decay_temp = membrane_potential * $signed({1'b0, BETA[7:0]});
        decay_potential = 24'(decay_temp >>> 7);  // Arithmetic right shift, truncate to 24 bits

        // Step 2: Calculate reset subtraction (threshold if prev spike, else 0)
        // `spike` register holds the spike value from the previous enabled cycle
        reset_subtract = spike ? THRESHOLD : 24'sd0;

        // Step 3: Compute next membrane: beta*mem + input - reset*threshold
        next_membrane = decay_potential + current_extended - reset_subtract;
    end

    // Sequential logic - update state on clock edge
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= 24'sd0;
            spike <= 1'b0;
        end else if (enable) begin
            // Update membrane potential
            membrane_potential <= next_membrane;

            // Generate spike if membrane >= threshold
            spike <= (next_membrane >= 24'($signed(THRESHOLD)));
        end
        // When enable=0, hold all state
    end

endmodule
