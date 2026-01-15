// Linear Layer Module (Sequential/Serial)
// Implements a fully connected layer: output = weights * input + bias
// Equivalent to nn.Linear in PyTorch
//
// Computes one output neuron per clock cycle (serial processing).
// Outputs are streamed one per cycle via output_current with corresponding output_idx.
//
// Fixed-point format: QS2.13 (16-bit signed, 2 integer bits, 13 fractional bits)
// Weights are stored in row-major order in a flattened array:
//   weights_flat[n*NUM_INPUTS + i] = weight for output n, input i
//
// Timing:
//   - Assert 'start' for one cycle with valid inputs
//   - Each of next NUM_OUTPUTS cycles: output_valid asserted with output_current and output_idx
//   - 'done' asserts on final output cycle
//   - Outputs remain valid until next 'start'

module linear_layer #(
    parameter NUM_INPUTS = 4,
    parameter NUM_OUTPUTS = 16,
    parameter DATA_WIDTH = 16,
    parameter FRAC_BITS = 13,
    parameter WEIGHTS_FILE = "weights.mem",
    parameter BIAS_FILE = "bias.mem"
) (
    input wire clk,
    input wire reset,
    input wire start,
    input wire signed [DATA_WIDTH-1:0] inputs [0:NUM_INPUTS-1],
    output logic signed [DATA_WIDTH-1:0] output_current,  // One current per cycle
    output logic [$clog2(NUM_OUTPUTS)-1:0] output_idx,   // Which neuron (0 to NUM_OUTPUTS-1)
    output logic output_valid,                            // Current output is valid
    output logic done
);

    // Index width for output counter
    localparam IDX_WIDTH = $clog2(NUM_OUTPUTS);

    // Flattened weights array: row-major order
    logic signed [DATA_WIDTH-1:0] weights_flat [0:NUM_OUTPUTS*NUM_INPUTS-1];
    logic signed [DATA_WIDTH-1:0] biases [0:NUM_OUTPUTS-1];

    // Load weights and biases from memory files
    initial begin
        $readmemh(WEIGHTS_FILE, weights_flat);
        $readmemh(BIAS_FILE, biases);
    end

    // Accumulator width: multiplication doubles bits, add extra for sum of NUM_INPUTS terms
    localparam ACCUM_WIDTH = 2 * DATA_WIDTH + $clog2(NUM_INPUTS + 1);

    // Saturation bounds
    localparam signed [DATA_WIDTH-1:0] MAX_VAL = {1'b0, {(DATA_WIDTH-1){1'b1}}};  // 32767 for 16-bit
    localparam signed [DATA_WIDTH-1:0] MIN_VAL = {1'b1, {(DATA_WIDTH-1){1'b0}}};  // -32768 for 16-bit

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        COMPUTING
    } state_t;

    state_t state;
    logic [IDX_WIDTH-1:0] neuron_idx;  // Counter for current output neuron

    // Latched inputs (held stable during computation)
    logic signed [DATA_WIDTH-1:0] inputs_latched [0:NUM_INPUTS-1];

    // Combinational computation of current neuron's output
    logic signed [ACCUM_WIDTH-1:0] accum;
    logic signed [ACCUM_WIDTH-1:0] scaled;
    logic signed [DATA_WIDTH-1:0] saturated;

    always_comb begin
        // Compute weighted sum for current neuron: Σ(input[i] * weight[neuron_idx][i])
        accum = 0;
        for (int i = 0; i < NUM_INPUTS; i++) begin
            accum = accum + inputs_latched[i] * weights_flat[neuron_idx * NUM_INPUTS + i];
        end

        // Scale back by FRAC_BITS (fixed-point multiply correction) and add bias
        scaled = (accum >>> FRAC_BITS) +
                 $signed({{(ACCUM_WIDTH-DATA_WIDTH){biases[neuron_idx][DATA_WIDTH-1]}},
                          biases[neuron_idx]});

        // Saturate to output range
        if (scaled > $signed({{(ACCUM_WIDTH-DATA_WIDTH){1'b0}}, MAX_VAL})) begin
            saturated = MAX_VAL;
        end else if (scaled < $signed({{(ACCUM_WIDTH-DATA_WIDTH){1'b1}}, MIN_VAL})) begin
            saturated = MIN_VAL;
        end else begin
            saturated = scaled[DATA_WIDTH-1:0];
        end
    end

    // FSM
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            neuron_idx <= '0;
            done <= 1'b0;
            output_valid <= 1'b0;
            output_current <= '0;
            output_idx <= '0;
            for (int i = 0; i < NUM_INPUTS; i++) begin
                inputs_latched[i] <= '0;
            end
        end else begin
            unique case (state)
                IDLE: begin
                    done <= 1'b0;
                    output_valid <= 1'b0;
                    if (start) begin
                        // Latch inputs and start computation
                        for (int i = 0; i < NUM_INPUTS; i++) begin
                            inputs_latched[i] <= inputs[i];
                        end
                        neuron_idx <= '0;
                        state <= COMPUTING;
                    end
                end

                COMPUTING: begin
                    // Output current result
                    output_current <= saturated;
                    output_idx <= neuron_idx;
                    output_valid <= 1'b1;

                    // Check if this is the last neuron
                    if (neuron_idx == IDX_WIDTH'(NUM_OUTPUTS - 1)) begin
                        done <= 1'b1;
                        state <= IDLE;
                    end else begin
                        neuron_idx <= neuron_idx + 1'b1;
                    end
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
