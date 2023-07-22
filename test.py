import torch
import tt_lib as ttl
from tests.python_api_testing.models.utility_functions_new import comp_pcc

device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
ttl.device.InitializeDevice(device)
host = ttl.device.GetHost()

torch.manual_seed(0)

def pt_reshape_layernorm_lastdim_norm(a, eps):
    a_r = a.reshape([1, 1, 1, a.numel()])
    layernorm_lastdim_norm_r = torch.nn.functional.layer_norm(
        a_r, a_r.shape[-1:], eps=eps
    )
    return layernorm_lastdim_norm_r.reshape(a.shape)


def tt_reshape_layernorm_lastdim_norm(a, eps):
    a_tt = ttl.tensor.Tensor(
        a.reshape(-1).tolist(),
        [1, 1, 1, a.numel()],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device)
    tt_layernorm_lastdim_norm_r = ttl.tensor.layernorm(a_tt, eps)
    return torch.Tensor(tt_layernorm_lastdim_norm_r.to(host).data()).reshape(a.shape)


def pt_transpose_layernorm_lastdim_norm(a, eps):
    a_t = a.transpose(1, 3)
    layernorm_lastdim_norm_t = torch.nn.functional.layer_norm(
        a_t, a_t.shape[-1:], eps=eps
    )
    return layernorm_lastdim_norm_t.transpose(1, 3)


def tt_groupnorm(a, eps):
    a_tt = ttl.tensor.Tensor(
        a.reshape(-1).tolist(),
        a.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(device)
    tt_groupnorm = ttl.tensor.groupnorm(a_tt, 1, eps)
    return torch.Tensor(tt_groupnorm.to(host).data()).reshape(a.shape)


a = torch.rand([1, 32, 4, 32])
eps = 1e-5

pt_groupnorm_out = torch.nn.functional.group_norm(a, 1, eps=eps)

pt_layernorm_3dim_norm_out = torch.nn.functional.layer_norm(a, a.shape[-3:], eps=eps)

pt_layernorm_reshape_lastdim_norm_out = pt_reshape_layernorm_lastdim_norm(a, eps)

tt_layernorm_reshape_lastdim_norm_out = tt_reshape_layernorm_lastdim_norm(a, eps)

pt_transpose_layernorm_lastdim_norm_out = pt_transpose_layernorm_lastdim_norm(a, eps)

tt_groupnorm_out = tt_groupnorm(a, eps)

ttl.device.CloseDevice(device)

print("1. pt groupnorm vs pt layernorm last 3 dim norm")
print(
    f"max_diff: {torch.amax(torch.abs(pt_groupnorm_out - pt_layernorm_3dim_norm_out))}"
)
print(comp_pcc(pt_groupnorm_out, pt_layernorm_3dim_norm_out))

print("2. pt groupnorm vs pt reshape layernorm last dim norm")
print(
    f"max_diff: {torch.amax(torch.abs(pt_groupnorm_out - pt_layernorm_reshape_lastdim_norm_out))}"
)
print(comp_pcc(pt_groupnorm_out, pt_layernorm_reshape_lastdim_norm_out))

print("3. pt groupnorm vs tt reshape layernorm last dim norm")
print(
    f"max_diff: {torch.amax(torch.abs(pt_groupnorm_out - tt_layernorm_reshape_lastdim_norm_out))}"
)
print(comp_pcc(pt_groupnorm_out, tt_layernorm_reshape_lastdim_norm_out))

print("4. pt groupnorm vs pt transpose layernorm last dim norm")
print(
    f"max_diff: {torch.amax(torch.abs(pt_groupnorm_out - pt_transpose_layernorm_lastdim_norm_out))}"
)
print(comp_pcc(pt_groupnorm_out, pt_transpose_layernorm_lastdim_norm_out))

print("5. pt groupnorm vs tt groupnorm (transpose layernorm last dim norm)")
print(f"max_diff: {torch.amax(torch.abs(pt_groupnorm_out - tt_groupnorm_out))}")
print(comp_pcc(pt_groupnorm_out, tt_groupnorm_out))
