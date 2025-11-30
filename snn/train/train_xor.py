import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from pathlib import Path

beta = 0.5

batch_size = 4
num_inputs = 2
num_hidden = 2
num_outputs = 1
num_steps = 10
num_epochs = 750

# Device setup
device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)

# XOR dataset - using static input encoding
# Each input pattern is presented as a constant value at every time step
# The SNN integrates this static signal over time through membrane dynamics
data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], device=device)
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Using Leaky neurons instead of Lapicque for cleaner surrogate gradient support
        # Leaky uses beta decay parameter and is designed for learning tasks
        # Lapicque models RC circuit dynamics - more biologically realistic but less common for training
        spike_grad = surrogate.fast_sigmoid()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


if __name__ == "__main__":
    # Initialize network, loss, and optimizer
    net = Net().to(device)

    # Using standard MSELoss instead of SF.mse_count_loss because:
    # - XOR is a regression task (binary outputs 0/1), not classification
    # - SF.mse_count_loss expects multi-class targets with one-hot encoding
    # - Plain MSE on spike counts is simpler and works well for this task
    # - Surrogate gradients still apply automatically through the Leaky neurons
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        spk_rec, mem_rec = net(data)

        # Compute spike count and compare to targets
        spike_count = spk_rec.sum(dim=0)  # Sum over time steps
        loss = loss_fn(spike_count, targets)

        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Test the network
    with torch.no_grad():
        spk_rec, mem_rec = net(data)
        spike_count = spk_rec.sum(dim=0)
        print("\nFinal Results:")
        print("Inputs -> Spike Count -> Target")
        for i in range(batch_size):
            print(
                f"{data[i].cpu().numpy()} -> {spike_count[i].item():.2f} -> {targets[i].item()}"
            )

    # Save model weights
    weights_dir = Path("./weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "snn_xor_weights.pth"
    torch.save(net.state_dict(), weights_path)
    print(f"\nModel weights saved to {weights_path}")
