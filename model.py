import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 
            2 * residual_channels, 
            kernel_size=3, 
            padding=dilation, 
            dilation=dilation
        )
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        output = self.dilated_conv(x)
        
        # Gated activation unit
        filter_out, gate_out = output.chunk(2, dim=1)
        output = torch.tanh(filter_out) * torch.sigmoid(gate_out)
        
        # Residual connection
        res_output = self.res_conv(output)
        input_cut = x 
        # If padding is correct, shapes should match. 
        # With padding=dilation and kernel_size=3, output length equals input length.
        
        output = (input_cut + res_output) * 0.7071 # Scale to keep variance stable
        
        # Skip connection
        skip_output = self.skip_conv(output)
        
        return output, skip_output

class WaveNetSourceSeparator(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 residual_channels=64, 
                 skip_channels=64, 
                 num_blocks=3, 
                 num_layers_per_block=10):
        super(WaveNetSourceSeparator, self).__init__()
        
        self.input_conv = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        
        self.residual_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for i in range(num_layers_per_block):
                dilation = 2 ** i
                self.residual_blocks.append(
                    ResidualBlock(residual_channels, skip_channels, dilation)
                )
                
        self.end_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: [Batch, Channels, Time]
        x = self.input_conv(x)
        
        skip_connections = []
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
            
        # Sum skip connections
        output = sum(skip_connections)
        
        # Output processing
        output = F.relu(output)
        output = self.end_conv1(output)
        output = F.relu(output)
        output = self.end_conv2(output)
        
        return output

if __name__ == "__main__":
    # Test model
    model = WaveNetSourceSeparator()
    x = torch.randn(1, 1, 16000)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
