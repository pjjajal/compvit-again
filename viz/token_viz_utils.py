from typing import List

import torch

# import argparse
# import matplotlib.pyplot as plt
# import seaborn as sns

# import torchvision.transforms.v2 as tvt
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from torch.utils.data import DataLoader
# from torchvision.datasets import ImageNet

# sns.set_theme("notebook")


# def parse_args():
#     parser = argparse.ArgumentParser("token compression visualization script")
#     parser.add_argument("--dataset", required=True, choices=["imagenet"])
#     parser.add_argument(
#         "--model",
#         required=True,
#         choices=["compvits14", "compvitb14", "compvitl14", "compvitg14"],
#     )
#     parser.add_argument("--checkpoint", required=True, type=str)
#     parser.add_argument("--image_idx", default=5000, type=int)


def hook_model(
    model,
    data_dict,
    module_names: List[str] = [
        "model.compressor.block_1.attn",
        "model.compressor.block_2.attn",
    ],
):
    """
    Attaches hooks to specified modules in the model and saves their outputs in the data_dict.

    Args:
        model: The model to attach hooks to.
        data_dict: A dictionary to store the outputs of the hooked modules.
        module_names: A list of module names to attach hooks to. Defaults to ["model.compressor.block_1.attn", "model.compressor.block_2.attn"].

    Returns:
        None
    """

    def hook(name):
        def _hook(module, input, outputs):
            outputs = [
                output.cpu().detach().numpy()
                for output in outputs
                if output is not None
            ]
            data_dict[name] = outputs

        return _hook

    for name, module in model.named_modules():
        if name in module_names:
            module.register_forward_hook(hook(name))


def attn_processing(attn_map, threshold=0.8):
    """
    Process attention map to extract top-p indices and values for each output token.

    Args:
        attn_map (torch.Tensor): Attention map of shape (head_num, out_tokens, in_tokens).
        threshold (float, optional): Threshold value for selecting top-p tokens. Defaults to 0.8.

    Returns:
        dict: A dictionary containing top-p indices and values for each output token, organized by head.

    """
    # attn_map should not have batch dimension.
    head_num, out_tokens, in_tokens = attn_map.shape

    # torch tensors because numpy can be annoying.
    attn_map = torch.tensor(attn_map)

    headwise_attn_ranks = {f"out_token:{out_token}": {} for out_token in range(out_tokens)}
    for head in range(head_num):
        sorted_values, sorted_indices = torch.sort(
            attn_map[head, :, :], descending=True, stable=True
        )

        # This is inefficient, but it works a bit better.
        boolean_mask = torch.zeros_like(sorted_values).bool()
        cum_sum = sorted_values.cumsum(dim=1)
        for out_token in range(out_tokens):
            for in_token in range(in_tokens):
                if cum_sum[out_token, in_token] <= threshold:
                    boolean_mask[out_token, in_token] = True
                else:
                    # Include the first token that exceeds the threshold.
                    boolean_mask[out_token, in_token] = True
                    break
        # By taking the cumsum, we can find how many tokens are needed to reach the attn threshold.
        # The xor operation is to handle the case where the first element is already greater than the threshold.
        # boolean_mask = sorted_values.cumsum(dim=1) < threshold
        # boolean_mask[:, 0] ^= ~(sorted_values.cumsum(dim=1) < threshold).any(dim=1)

        # We can now use the boolean mask to index the sorted indices and find their corresponding attn values.
        topp_indicies = ((sorted_indices + 1) * boolean_mask) - 1
        topp_values = sorted_values.clone()
        topp_values[~boolean_mask] = float(
            "nan"
        )  # this makes it easier to ignore the values that are not in the top-p.

        # We can now store the top-p indices and values for each output token.
        for out_token in range(out_tokens):
            headwise_attn_ranks[f"out_token:{out_token}"][f"head:{head}"] = {
                "topp_indices": topp_indicies[
                    out_token, topp_indicies[out_token, :] != -1
                ].tolist(),
                "topp_values": topp_values[
                    out_token, topp_values[out_token, :].isfinite()
                ].tolist(),
            }
    return headwise_attn_ranks
