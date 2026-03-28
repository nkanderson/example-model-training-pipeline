// Fractional-Order Leaky Integrate-and-Fire (LIF) Neuron Module
// Implements a fractional-order spiking neuron using Grünwald-Letnikov approximation
// Drop-in replacement for lif.sv with matching interface
//
// Membrane dynamics (from fractional_lif.py):
//   V[n] = (I[n] - C * Σ_{k=1}^{H-1} g_k * V[n-k]) / (C + λ)
//   where C = 1 / dt^α
//
// Spike generation: spike[t] = (mem[t] >= threshold)
// Reset: subtract threshold from membrane on spike (reset_delay=True style)
//
// Relationship between standard LIF beta and fractional lambda:
//   λ = (1 - β) / β
//   For β = 0.9 (default): λ = 0.1 / 0.9 ≈ 0.111
//   This provides approximate equivalence in membrane decay behavior.
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)

module fractional_lif #(
    // Standard LIF parameters (match lif.sv interface)
    parameter THRESHOLD = 8192,              // Spike threshold (1.0 in QS2.13)
    parameter DATA_WIDTH = 16,
    parameter MEMBRANE_WIDTH = 24,
    
    // Fractional-order parameters
    parameter HISTORY_LENGTH = 64,           // Number of past values for GL approximation
    parameter COEFF_WIDTH = 16,              // GL coefficient magnitude width (QU1.15 unsigned)
    parameter COEFF_FRAC_BITS = 15,          // Fractional bits in coefficients
    
    // Precomputed constants (from generate_coefficients.py)
    // For α=0.5, dt=1.0, β=0.9 (→ λ=0.111): C=1.0, denom=1.111
    // Relationship: λ = (1 - β) / β
    // C_SCALED in Q8.8: 1.0 * 256 = 256
    parameter [15:0] C_SCALED = 16'd256,
    // INV_DENOM in Q0.16: 1/1.111 ≈ 0.9 * 65536 ≈ 58982
    parameter [15:0] INV_DENOM = 16'd58982,
    
    // Coefficient file (GL coefficient magnitudes |g_1| to |g_{H-1}|)
    parameter GL_COEFF_FILE = "gl_coefficients.mem"
) (
    input wire clk,
    input wire reset,
    input wire clear,                                    // Synchronous clear for new inference
    input wire enable,                                   // Process one timestep
    input wire signed [DATA_WIDTH-1:0] current,          // Input current (QS2.13 format)
    output logic spike_out,                              // Spike output this timestep
    output logic signed [MEMBRANE_WIDTH-1:0] membrane_out // Membrane potential after update
);

    // Address width for history indexing
    localparam integer ADDR_WIDTH = $clog2(HISTORY_LENGTH);

    // Internal state
    logic signed [MEMBRANE_WIDTH-1:0] membrane_potential;  // Membrane potential
    logic spike_prev;                                      // Previous spike for reset delay

    // History buffer (circular buffer for past membrane potentials)
    logic signed [MEMBRANE_WIDTH-1:0] history_buffer [0:HISTORY_LENGTH-1];
    logic [ADDR_WIDTH-1:0] history_ptr;      // Points to oldest value (next write location)

    // GL coefficient magnitudes (unsigned, loaded from file)
    // |g_k| for k=1 to HISTORY_LENGTH-1 (g_0=1 is implicit and not stored)
    // Assumes 0 < alpha <= 1, where g_k (k>=1) are non-positive.
    logic [COEFF_WIDTH-1:0] gl_coeffs_mag [0:HISTORY_LENGTH-2];
    
    // Load pre-computed GL coefficients from memory file
    initial begin
        $readmemh(GL_COEFF_FILE, gl_coeffs_mag, 0, HISTORY_LENGTH-2);
    end

    // Intermediate computation signals
    logic signed [MEMBRANE_WIDTH-1:0] next_membrane;
    logic signed [MEMBRANE_WIDTH-1:0] current_extended;
    logic signed [MEMBRANE_WIDTH-1:0] reset_subtract;
    logic next_spike;
    
    // History magnitude-weighted sum computation (Σ |g_k| * V[n-k] for k=1 to H-1)
    // This needs to be pipelined for large HISTORY_LENGTH, but for now use combinational
    logic signed [47:0] history_sum;  // Wide accumulator for sum of products

    // Combinational logic to compute next membrane potential
    // Implements fractional LIF using |g_k| with 0<alpha<=1 sign property:
    // V[n] = (I[n] + C * Σ |g_k| * V[n-k]) / (C + λ)
    always_comb begin
        logic [ADDR_WIDTH-1:0] hist_idx;
        logic [ADDR_WIDTH-1:0] k_plus_1;
        logic signed [MEMBRANE_WIDTH-1:0] hist_val;
        logic [COEFF_WIDTH-1:0] coeff_mag;
        logic signed [MEMBRANE_WIDTH+COEFF_WIDTH:0] product;
        logic signed [63:0] scaled_history;
        logic signed [MEMBRANE_WIDTH-1:0] numerator;
        logic signed [MEMBRANE_WIDTH+15:0] scaled_result;

        // Default assignments to prevent latches
        current_extended = '0;
        reset_subtract = '0;
        next_membrane = '0;
        next_spike = 1'b0;
        history_sum = '0;

        // Sign-extend current from DATA_WIDTH to MEMBRANE_WIDTH
        current_extended = {{(MEMBRANE_WIDTH-DATA_WIDTH){current[DATA_WIDTH-1]}}, current};

        // Step 1: Compute history sum Σ_{k=1}^{H-1} |g_k| * V[n-k]
        // history_ptr points to oldest value, so most recent V[n-1] is at (history_ptr - 1)
        for (int k = 0; k < HISTORY_LENGTH - 1; k++) begin
            // k=0 in loop corresponds to |g_1| * V[n-1], k=1 to |g_2| * V[n-2], etc.
            k_plus_1 = ADDR_WIDTH'(k + 1);

            // Calculate circular buffer index for V[n-k-1]
            // history_ptr is where next write goes (oldest), so V[n-1] is at history_ptr-1
            if (history_ptr >= k_plus_1) begin
                hist_idx = history_ptr - k_plus_1;
            end else begin
                hist_idx = history_ptr + ADDR_WIDTH'(HISTORY_LENGTH) - k_plus_1;
            end

            hist_val = history_buffer[hist_idx];
            coeff_mag = gl_coeffs_mag[k];
            product = $signed({1'b0, coeff_mag}) * hist_val;

            // Accumulate (sign-extend product to accumulator width)
            history_sum = history_sum + {{(48-(MEMBRANE_WIDTH+COEFF_WIDTH+1)){product[MEMBRANE_WIDTH+COEFF_WIDTH]}}, product};
        end

        // Step 2: Calculate reset subtraction (threshold if prev spike, else 0)
        // reset_delay=True: subtract threshold one cycle after spike
        reset_subtract = spike_prev ? MEMBRANE_WIDTH'($signed(THRESHOLD)) : '0;

        // Step 3: Compute V[n] = (I[n] + C * history_sum) / (C + λ)
        // Using precomputed C_SCALED (Q8.8) and INV_DENOM (Q0.16)
        // 
        // Numerator = current_extended + (C_SCALED * history_sum) >> 8
        // Then multiply by INV_DENOM and shift to get final membrane
        begin
            // Scale history_sum by C (right shift by 8 for Q8.8, and by COEFF_FRAC_BITS for coeff format)
            scaled_history = (C_SCALED * history_sum) >>> (8 + COEFF_FRAC_BITS);

            // Numerator = I[n] + C * Σ |g_k| * V[n-k]
            numerator = current_extended + MEMBRANE_WIDTH'(scaled_history);

            // Divide by (C + λ) via multiplication by 1/(C+λ) = INV_DENOM
            // INV_DENOM is Q0.16, so result needs >> 16
            scaled_result = numerator * $signed({1'b0, INV_DENOM});

            // Apply reset subtraction and extract final membrane value
            next_membrane = MEMBRANE_WIDTH'(scaled_result >>> 16) - reset_subtract;
        end

        // Step 4: Generate spike if membrane >= threshold
        next_spike = (next_membrane >= MEMBRANE_WIDTH'($signed(THRESHOLD)));
    end

    // Sequential logic - update state on clock edge
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
            history_ptr <= '0;
            
            // Clear history buffer
            for (int i = 0; i < HISTORY_LENGTH; i++) begin
                history_buffer[i] <= '0;
            end
        end else if (clear) begin
            // Synchronous clear for new inference
            membrane_potential <= '0;
            spike_prev <= 1'b0;
            spike_out <= 1'b0;
            membrane_out <= '0;
            history_ptr <= '0;
            
            // Clear history buffer
            for (int i = 0; i < HISTORY_LENGTH; i++) begin
                history_buffer[i] <= '0;
            end
        end else if (enable) begin
            // Process one timestep
            
            // Store current membrane in history before updating
            history_buffer[history_ptr] <= membrane_potential;
            history_ptr <= (history_ptr == ADDR_WIDTH'(HISTORY_LENGTH - 1)) ? '0 : history_ptr + 1'b1;
            
            // Update membrane potential
            membrane_potential <= next_membrane;
            spike_prev <= next_spike;
            spike_out <= next_spike;
            membrane_out <= next_membrane;
        end
        // else: hold state when not enabled
    end

endmodule
