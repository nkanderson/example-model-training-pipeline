#include "Vnn_tb.h"
#include "verilated.h"
#include "verilated_vcd_c.h"

int main(int argc, char **argv) {
  // Create logs/ directory in case we have traces to put under it
  Verilated::mkdir("logs");

  // Construct a VerilatedContext to hold simulation time, etc.
  const std::unique_ptr<VerilatedContext> contextp{new VerilatedContext};

  // Set commandline arguments
  contextp->commandArgs(argc, argv);

  // Construct the Verilated model
  const std::unique_ptr<Vnn_tb> top{new Vnn_tb{contextp.get()}};

  // Set up tracing
  contextp->traceEverOn(true);
  VerilatedVcdC *tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("nn_tb.vcd");

  // Simulate until $finish
  while (!contextp->gotFinish()) {
    // Evaluate model
    top->eval();

    // Dump trace data for this cycle
    tfp->dump(contextp->time());

    // Advance time
    contextp->timeInc(1);
  }

  // Final model cleanup
  tfp->close();
  top->final();

  return 0;
}
