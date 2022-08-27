import torch


def SimpleEncoder(img_batch):
    img_stack = []
    for img in img_batch:
        img_stack.append(img.view(-1))
    return torch.stack(img_stack)
