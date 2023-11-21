from abc import ABC, abstractmethod
from typing import List


# Base class for OP
class TTPyOp(ABC):
    # Generate op config variabes and tensors
    @abstractmethod
    def set_op_configs(self):
        pass

    # Construct pytorch tensors for op weights and bias. Moves those tensors to device
    @abstractmethod
    def set_op_weights_biases(self, weight_tensor: List, bias_tensor: List):
        pass

    # Return stats on op's L1 buffers
    @abstractmethod
    def get_l1_buffer_stats(self):
        pass

    @abstractmethod
    def run_forward(self):
        pass
