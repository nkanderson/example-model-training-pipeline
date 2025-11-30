// Testbench for SNN Neural Network
// Tests XOR functionality with spike counting over timesteps

`timescale 1ns/1ps

module neural_network_tb;

    // Clock and reset
    reg clk;
    reg reset;
    
    // DUT signals
    reg start;
    reg [1:0] inputs;
    wire done;
    wire [7:0] spike_count;
    
    // Test variables
    integer test_num;
    integer cycle_count;
    
    // Instantiate DUT
    neural_network dut (
        .clk(clk),
        .reset(reset),
        .start(start),
        .inputs(inputs),
        .done(done),
        .spike_count(spike_count)
    );
    
    // Clock generation (10ns period = 100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    // Test procedure
    initial begin
        // Generate VCD waveform file
        $dumpfile("dump.vcd");
        $dumpvars(0, neural_network_tb);
        
        $display("========================================");
        $display("SNN Neural Network XOR Test");
        $display("========================================");
        
        // Initialize
        reset = 1;
        start = 0;
        inputs = 2'b00;
        test_num = 0;
        
        // Hold reset
        repeat(5) @(posedge clk);
        reset = 0;
        repeat(2) @(posedge clk);
        
        // Test case 0,0 -> expect low spike count
        run_test(2'b00, "0,0", "low");
        
        // Test case 0,1 -> expect high spike count
        // inputs[0]=0, inputs[1]=1 means bit pattern 2'b10
        run_test(2'b10, "0,1", "high");
        
        // Test case 1,0 -> expect high spike count
        // inputs[0]=1, inputs[1]=0 means bit pattern 2'b01
        run_test(2'b01, "1,0", "high");
        
        // Test case 1,1 -> expect low spike count
        run_test(2'b11, "1,1", "low");
        
        $display("\n========================================");
        $display("All tests completed!");
        $display("========================================");
        
        $finish;
    end
    
    // Task to run a single test
    task run_test;
        input [1:0] test_inputs;
        input [63:0] input_str;
        input [63:0] expected;
        begin
            test_num = test_num + 1;
            
            $display("\n--- Test %0d: [%s] ---", test_num, input_str);
            $display("Expected: %s spike count", expected);
            
            // Set inputs
            inputs = test_inputs;
            @(posedge clk);
            
            // Start processing
            start = 1;
            @(posedge clk);
            start = 0;
            
            // Wait for done signal (with timeout)
            cycle_count = 0;
            while (!done && cycle_count < 50) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end
            
            if (!done) begin
                $display("ERROR: Timeout waiting for done signal");
                $finish;
            end
            
            $display("Completed in %0d cycles", cycle_count);
            $display("Spike count: %0d", spike_count);
            
            // Validate results
            if (expected == "high") begin
                if (spike_count >= 1) begin
                    $display("PASS: High spike count detected");
                end else begin
                    $display("FAIL: Expected high spike count (>=3), got %0d", spike_count);
                end
            end else begin // "low"
                if (spike_count == 0) begin
                    $display("PASS: Low spike count detected");
                end else begin
                    $display("FAIL: Expected low spike count (<=2), got %0d", spike_count);
                end
            end
            
            // Reset for next test
            reset = 1;
            repeat(5) @(posedge clk);
            reset = 0;
            repeat(2) @(posedge clk);
        end
    endtask

endmodule
