# prototype

## cylinder
Cylinder.__init__(in_features: int, depth: int, interlayer: list = None, pipe_head: list = None, pipe_end: list = None)

## pyramid
Pyramid.__init__(in_features: int, out_features: int, depth: int, interlayer: list = None, pipe_head: list = None, pipe_end: list = None)

## ohlcv_receptor
OHLCVReceptor.__init__(hidden: int = 2, side_dim: int = 2, hidden_v: int = 4)
OHLCVReceptor.forward(hocl: torch.Tensor, v: torch.Tensor) -> torch.Tensor
