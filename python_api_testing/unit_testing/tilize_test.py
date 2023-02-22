import numpy as np 

import ll_buda_bindings.ll_buda_bindings._C as _C

def run_tilize_test():
    nt = 1
    shape = [32, 32 * nt]
    a = _C.tensor.Tensor(
        [float(i) for i in range(1024 * nt)],
        [1, 1] + shape,
        _C.tensor.DataFormat.FLOAT32,
        _C.tensor.Layout.ROW_MAJOR,
        device
    )

    b = _C.tensor.tilize(a)

    c = np.array(b.to(host).data(), dtype=int).reshape(*shape)

if __name__ == "__main__":
    device = _C.device.CreateDevice(_C.device.Arch.GRAYSKULL, 0)
    _C.device.InitializeDevice(device)
    host = _C.device.GetHost()
    run_tilize_test()
    _C.device.CloseDevice(device)