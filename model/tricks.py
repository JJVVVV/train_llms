import torch
import torch.nn.functional as F


def generate_spherical_vector(size: tuple, device, dtype, r: torch.Tensor = torch.tensor(1)):
    # 生成均匀分布在半径为 r 的超球面上的向量
    v = torch.randn(size, device=device, dtype=dtype)
    v = F.normalize(v, p=2, dim=-1)
    return v.mul(r)


def generate_ball_vector(size: tuple, device, r: torch.Tensor = torch.tensor(1)):
    # 生成均匀分布在半径为 r 的超球体内的向量
    size = size[:-1] + (size[-1] + 2,)
    v = torch.randn(size, device=device)
    v = F.normalize(v, p=2, dim=-1)
    return v.mul(r)[..., :-2]


# shift1
def shift_embeddings(input_embs: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
    # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
    with torch.no_grad():
        r = torch.sqrt(input_embs.mul(input_embs).sum(dim=-1, keepdim=True)) * alpha
        return input_embs + generate_spherical_vector(input_embs.size(), input_embs.device, input_embs.dtype, r)


# def shift_embeddings(input_embs: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
#     # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
#     r = torch.sqrt(input_embs.mul(input_embs).sum(dim=-1, keepdim=True)) * alpha
#     return input_embs + generate_spherical_vector(input_embs.size(), input_embs.device, r)


# # shift4
# def shift_embeddings(input_embs: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
#     # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
#     with torch.no_grad():
#         r = torch.sqrt(input_embs.mul(input_embs).sum(dim=-1, keepdim=True))
#         scale = (r + alpha) / r
#         center = input_embs * scale
#         return center + generate_spherical_vector(center.size(), center.device)


# # shift2
# def shift_embeddings(input_embs: torch.Tensor, alpha: float = 1) -> torch.Tensor:
#     # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
#     with torch.no_grad():
#         return input_embs + generate_spherical_vector(input_embs.size(), input_embs.device, alpha)


# # shift3
# def shift_embeddings(input_embs: torch.Tensor, alpha: float = 0.2) -> torch.Tensor:
#     # 为每个 token 的 embedding 添加随机偏移, 平移量为原 embedding 向量的模的 alpha 倍
#     with torch.no_grad():
#         center = input_embs * (1 - alpha)
#         r = torch.sqrt(input_embs.mul(input_embs).sum(dim=-1, keepdim=True)) * alpha
#         return center + generate_spherical_vector(center.size(), center.device, r)
