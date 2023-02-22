import numpy as np 

import ll_buda_bindings.ll_buda_bindings._C as _C

def run_untilize_test():
    a = _C.tensor.Tensor(
        [float(i) for i in range(1024)],
        [1, 1, 32, 32],
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.TILE,
        device
    )

    b = _C.tensor.untilize(a)

    c = np.array(b.to(host).data(), dtype=int).reshape(32, 32)
    print(c)
    breakpoint()

if __name__ == "__main__":
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    run_untilize_test()
    _C.device.CloseDevice(device)