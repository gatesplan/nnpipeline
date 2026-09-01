# prototype

## cylinder
Cylinder.__init__(in_features: int, depth: int, interlayer: list = None, pipe_head: list = None, pipe_end: list = None)

## pyramid
Pyramid.__init__(in_features: int, out_features: int, depth: int, interlayer: list = None, pipe_head: list = None, pipe_end: list = None)

## ohlcv_receptor
OHLCVReceptor.__init__(hidden: int = 2, side_dim: int = 2, hidden_v: int = 4)
OHLCVReceptor.forward(hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor

## receptor_bundle
ReceptorBundle.__init__(children: list, aggregator: nn.Module)
ReceptorBundle.forward(hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor
ReceptorBundle.n_leaves: int

## decay_bank
DecayBank.__init__(half_lives: tuple = (2.0, 8.0, 32.0), learnable: bool = True, include_diffs: bool = True, bias_correction: bool = True)
DecayBank.forward(e: torch.Tensor, return_sequence: bool = False) -> torch.Tensor
DecayBank.n_scales: int
DecayBank.out_scales: int
DecayBank.lambdas: torch.Tensor
DecayBank.half_lives: torch.Tensor
