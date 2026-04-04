import math
import torch
import torch.nn as nn
from torch import Tensor


# ==========================================================
# LstmCell  (mirrors model.rs LstmCell)
# ==========================================================
#
# LSTM equations:
#   f_t = σ(W_f·[h_{t-1}, x_t] + b_f)        # Forget gate
#   i_t = σ(W_i·[h_{t-1}, x_t] + b_i)        # Input gate
#   g_t = tanh(W_g·[h_{t-1}, x_t] + b_g)     # Candidate cell state
#   o_t = σ(W_o·[h_{t-1}, x_t] + b_o)        # Output gate
#
#   c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t          # New cell state
#   h_t = o_t ⊙ tanh(c_t)                     # New hidden state
#
# Gate ordering matches Rust: [i, f, g, o]  (split along dim 1)

class LstmCell(nn.Module):
    """
    Manual LSTM cell with layer normalization — exactly mirrors Rust LstmCell.

    Attributes
    ----------
    hidden_size : int
    weight_ih   : Linear(input_size  → 4 * hidden_size)   Xavier normal init
    weight_hh   : Linear(hidden_size → 4 * hidden_size)   Xavier normal init
    norm_x      : LayerNorm(4 * hidden_size)  — normalizes gate pre-acts from input
    norm_h      : LayerNorm(hidden_size)      — normalizes hidden state
    norm_c      : LayerNorm(hidden_size)      — normalizes cell state
    dropout     : Dropout(p)
    """

    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        # Weight matrices for input → gates  and  hidden → gates
        self.weight_ih = nn.Linear(input_size, 4 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 4 * hidden_size)

        # Layer normalizations
        self.norm_x = nn.LayerNorm(4 * hidden_size)
        self.norm_h = nn.LayerNorm(hidden_size)
        self.norm_c = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """
        Xavier normal initialization for weight matrices.
        Forget-gate bias (index hidden_size : 2*hidden_size) initialized to 1.0.
        Mirrors Rust's Initializer::XavierNormal and forget-gate bias trick.
        """
        for linear in (self.weight_ih, self.weight_hh):
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            # Set forget-gate bias to 1.0  (slice [H : 2H])
            with torch.no_grad():
                linear.bias[self.hidden_size : 2 * self.hidden_size].fill_(1.0)

    def init_state(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return zero (h, c) state — mirrors LstmCell::init_state."""
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        c = torch.zeros(batch_size, self.hidden_size, device=device)
        return h, c

    def forward(
        self,
        x: Tensor,                    # (batch, input_size)
        state: tuple[Tensor, Tensor], # ((batch, H), (batch, H))
    ) -> tuple[Tensor, Tensor]:
        """
        Single LSTM step.

        Returns new (h_t, c_t) — mirrors Rust LstmCell::forward returning LstmState.
        """
        h_prev, c_prev = state

        # Gate pre-activations from input and previous hidden state
        gates_x = self.weight_ih(x)          # (batch, 4H)
        gates_h = self.weight_hh(h_prev)     # (batch, 4H)

        # Apply layer norm to input contribution, then add hidden contribution
        # Matches Rust: norm_x on gates_x, then gates = gates_x + gates_h
        gates = self.norm_x(gates_x) + gates_h  # (batch, 4H)

        # Split into individual gates: [i, f, g, o]  (same order as Rust)
        i_gate, f_gate, g_gate, o_gate = gates.chunk(4, dim=1)

        i_t = torch.sigmoid(i_gate)
        f_t = torch.sigmoid(f_gate)
        g_t = torch.tanh(g_gate)
        o_t = torch.sigmoid(o_gate)

        # Update cell and hidden states
        c_t = f_t * c_prev + i_t * g_t
        c_t = self.norm_c(c_t)

        h_t = o_t * torch.tanh(c_t)
        h_t = self.norm_h(h_t)

        h_t = self.dropout(h_t)

        return h_t, c_t


# ==========================================================
# StackedLstm  (mirrors model.rs StackedLstm)
# ==========================================================

class StackedLstm(nn.Module):
    """
    Multiple LstmCells stacked layer-by-layer.
    No dropout is applied on the last layer (mirrors Rust behavior).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        cells = []
        for i in range(num_layers):
            in_sz = input_size if i == 0 else hidden_size
            # Last layer has no dropout
            layer_dropout = 0.0 if i == num_layers - 1 else dropout
            cells.append(LstmCell(in_sz, hidden_size, layer_dropout))

        self.layers = nn.ModuleList(cells)
        self.hidden_size = hidden_size

    def forward(
        self,
        x: Tensor,                              # (batch, seq_len, input_size)
        states: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """
        Process full sequence through all stacked layers.

        Returns
        -------
        output : Tensor  (batch, seq_len, hidden_size)
        states : list[(h, c)] one per layer
        """
        batch_size, seq_length, _ = x.shape
        device = x.device

        # Initialize states to zeros if not provided
        if states is None:
            states = [
                cell.init_state(batch_size, device) for cell in self.layers
            ]

        layer_outputs: list[Tensor] = []

        for t in range(seq_length):
            # x_t: (batch, input_size) for this time step
            input_t = x[:, t, :]

            for i, cell in enumerate(self.layers):
                h_new, c_new = cell(input_t, states[i])
                states[i] = (h_new, c_new)
                input_t = h_new  # next layer's input is this layer's hidden state

            layer_outputs.append(input_t)  # last layer's hidden state at time t

        # Stack along sequence dimension → (batch, seq_len, hidden_size)
        output = torch.stack(layer_outputs, dim=1)

        return output, states


# ==========================================================
# LstmNetwork  (mirrors model.rs LstmNetwork)
# ==========================================================

class LstmNetwork(nn.Module):
    """
    Full LSTM network with optional bidirectional support.

    In bidirectional mode:
    1. Forward  LSTM processes sequence l→r
    2. Backward LSTM processes reversed sequence, output is flipped back
    3. Outputs are concatenated along feature dim
    4. Dropout → Linear on last-timestep hidden state
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 32,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        self.bidirectional = bidirectional

        self.stacked_lstm = StackedLstm(input_size, hidden_size, num_layers, dropout)

        if bidirectional:
            self.reverse_lstm: StackedLstm | None = StackedLstm(
                input_size, hidden_size, num_layers, dropout
            )
            fc_in = 2 * hidden_size
        else:
            self.reverse_lstm = None
            fc_in = hidden_size

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(fc_in, output_size)

    def forward(
        self,
        x: Tensor,                              # (batch, seq_len, input_size)
        states: list[tuple[Tensor, Tensor]] | None = None,
    ) -> Tensor:                                # (batch, output_size)
        """
        Forward pass.  Uses the last-timestep hidden state for prediction.
        """
        seq_length = x.shape[1]

        # Forward direction
        output, _ = self.stacked_lstm(x, states)   # (batch, seq_len, H)

        if self.reverse_lstm is not None:
            # Reverse the sequence along the time dimension
            x_rev = x.flip(dims=[1])
            rev_output, _ = self.reverse_lstm(x_rev, None)   # (batch, seq_len, H)
            # Flip back so positions align with the forward output
            rev_output = rev_output.flip(dims=[1])
            # Concatenate on the feature dimension → (batch, seq_len, 2H)
            output = torch.cat([output, rev_output], dim=2)

        # Dropout before final projection
        output = self.dropout(output)

        # Use the last timestep's hidden state for regression output
        last = output[:, seq_length - 1, :]   # (batch, H or 2H)

        return self.fc(last)                   # (batch, output_size)
