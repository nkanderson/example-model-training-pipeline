// SNN Neural Network Module
// Implements a 2-2-1 spiking neural network for XOR
// 2 input neurons -> 2 hidden LIF neurons -> 1 output LIF neuron

module neural_network #(
    parameter NUM_TIMESTEPS = 12  // NOTE: Training uses 10 timesteps
) (
    input wire clk,
    input wire reset,
    input wire start,              // Start processing
    input wire [1:0] inputs,       // Binary inputs [x0, x1]
    output reg done,               // Processing complete
    output reg [7:0] spike_count   // Total output spikes
);

    // State machine
    typedef enum logic [1:0] {
        IDLE,
        PROCESSING,
        DONE_STATE
    } state_t;
    
    state_t state, next_state;
    reg [4:0] timestep_counter;  // 0-24 (5 bits for 25 timesteps)
    
    // Weights and biases (signed 8-bit)
    reg signed [7:0] hidden_weights [0:1][0:1];  // [neuron][input]
    reg signed [7:0] hidden_biases [0:1];
    reg signed [7:0] output_weights [0:1];       // [hidden_input]
    reg signed [7:0] output_bias [0:0];          // Single element array for $readmemh
    
    // Temporary arrays for loading weights (Verilator workaround)
    reg signed [7:0] hidden_w0_temp [0:1];
    reg signed [7:0] hidden_w1_temp [0:1];
    
    // LIF neuron signals
    wire hidden_spike [0:1]; // hidden layer outputs
    reg signed [7:0] hidden_current [0:1]; // hidden layer inputs
    wire output_spike; // output layer output
    reg signed [7:0] output_current; // output layer input
    
    // Enable signal for LIF neurons - only active during PROCESSING state
    wire enable;
    assign enable = (state == PROCESSING);
    
    // Load weights from memory files
    initial begin
        // Load hidden weights into temporary arrays
        $readmemh("weights/hidden_neuron_0.mem", hidden_w0_temp);
        $readmemh("weights/hidden_neuron_1.mem", hidden_w1_temp);
        $readmemh("weights/hidden_bias.mem", hidden_biases);
        $readmemh("weights/output_neuron.mem", output_weights);
        $readmemh("weights/output_bias.mem", output_bias);
        
        // Copy to multi-dimensional array
        hidden_weights[0][0] = hidden_w0_temp[0];
        hidden_weights[0][1] = hidden_w0_temp[1];
        hidden_weights[1][0] = hidden_w1_temp[0];
        hidden_weights[1][1] = hidden_w1_temp[1];
    end
    
    // Instantiate LIF neurons with matching parameters (threshold=1.0 -> 100 to match weight scale)
    lif #(.THRESHOLD(100)) hidden_lif_0 (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .current(hidden_current[0]),
        .spike(hidden_spike[0])
    );
    
    lif #(.THRESHOLD(100)) hidden_lif_1 (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .current(hidden_current[1]),
        .spike(hidden_spike[1])
    );
    
    lif #(.THRESHOLD(100)) output_lif (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .current(output_current),
        .spike(output_spike)
    );
    
    // State update transition on clock edge
    always_ff @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= IDLE;
            timestep_counter <= 5'd0;
            spike_count <= 8'd0;
        end else begin
            state <= next_state;
            
            unique case (state)
                IDLE: begin
                    if (start) begin
                        timestep_counter <= 5'd0;
                        spike_count <= 8'd0;
                    end
                end
                
                PROCESSING: begin
                    timestep_counter <= timestep_counter + 1;
                    // Accumulate output spikes
                    if (output_spike) begin
                        spike_count <= spike_count + 1;
                    end
                end
                
                DONE_STATE: begin
                    // Hold spike_count
                end
            endcase
        end
    end
    
    // Next-state logic
    always_comb begin
        // Stay in current state by default
        next_state = state;
        done = 1'b0;
        
        unique case (state)
            IDLE: begin
                if (start) begin
                    next_state = PROCESSING;
                end
            end
            
            PROCESSING: begin
                if (timestep_counter == NUM_TIMESTEPS - 1) begin
                    next_state = DONE_STATE;
                end
            end
            
            DONE_STATE: begin
                done = 1'b1;
                // Stay in DONE until reset or new start
                if (start) begin
                    next_state = PROCESSING;
                end
            end
        endcase
    end
    
    // Compute currents for hidden layer (conditional accumulation)
    always_comb begin
        integer i;
        for (i = 0; i < 2; i = i + 1) begin
            hidden_current[i] = hidden_biases[i];
            if (inputs[0]) begin
                hidden_current[i] = hidden_current[i] + hidden_weights[i][0];
            end
            if (inputs[1]) begin
                hidden_current[i] = hidden_current[i] + hidden_weights[i][1];
            end
        end
    end
    
    // Compute current for output layer (conditional accumulation)
    always_comb begin
        output_current = output_bias[0];
        if (hidden_spike[0]) begin
            output_current = output_current + output_weights[0];
        end
        if (hidden_spike[1]) begin
            output_current = output_current + output_weights[1];
        end
    end

endmodule
