# [IROS 2024] Meta-Initialized-Depth
[IROS 2024] Boosting Generalizability towards Zero-Shot Cross-Dataset Single-Image Indoor Depth by Meta-Initialization

## :dolphin: The following shows the main recipe for training meta-initialization for depth estimation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from timm.models import create_model
import argparse


class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()

    def forward(self, pred_map):
        def gradient(pred):
            D_dy = pred[:, :, 1:] - pred[:, :, :-1]
            D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
            return D_dx, D_dy

        if type(pred_map) not in [tuple, list]:
            pred_map = [pred_map]

        loss = 0
        weight = 1.

        for scaled_map in pred_map:
            dx, dy = gradient(scaled_map)
            dx2, dxdy = gradient(dx)
            dydx, dy2 = gradient(dy)
            loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
            weight /= 2.3  # don't ask me why it works better
        return loss

class Trainer:
    def __init__(self, options):        
        self.opt = options
        self.models["encoder"] = create_model('convnext_base', pretrained=True)
        self.models["encoder"] = create_model('convnext_small', pretrained=True)
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["decoder"] = DepthDecoder(...)
        self.models["decoder"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.SGD(self.parameters_to_train, self.opt.learning_rate)
        self.sup_model_optimizer = optim.Adam(self.parameters_to_train, self.opt.sup_learning_rate)
        self.sup_model_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.sup_model_optimizer, self.opt.iterations, eta_min=1e-6) 
        self.mse_loss = torch.nn.MSELoss()
        self.smoothness_loss = SmoothLoss()

    def inner_loop(self, inner_encoder, inner_depth, optim, inner_steps, imgs=None, gt=None):
        """
        train the inner model for a specified number of iterations
        """

        for step in range(inner_steps):
            # Your dataloader
            inputs = next(self.train_dataloader_iter)
            imgs = inputs["image"].cuda()
            gt = inputs["depth_gt"].cuda()

            optim.zero_grad()
            features = inner_encoder.forward_features(imgs)
            depth = inner_depth(features)
            loss = self.mse_loss(depth, gt) + 0.001 * self.smoothness_loss(depth)
            loss.backward()
            optim.step()

    def run_epoch_reptile(self):
        self.model_optimizer.zero_grad()
        inner_encoder = copy.deepcopy(self.models["encoder"])
        inner_depth = copy.deepcopy(self.models["decoder"])
        inner_optim = torch.optim.SGD(list(inner_encoder.parameters())+list(inner_depth.parameters()), self.opt.inner_lr)
        torch.nn.utils.clip_grad_value_(list(inner_encoder.parameters())+list(inner_depth.parameters()), 1.0)
        self.inner_loop(inner_encoder, inner_depth, inner_optim, self.opt.inner_steps)

        with torch.no_grad():
            for meta_param_enc, inner_param_enc in zip(self.models["encoder"].parameters(), inner_encoder.parameters()):
                meta_param_enc.grad = meta_param_enc - inner_param_enc
            for meta_param_dec, inner_param_dec in zip(self.models["decoder"].parameters(), inner_depth.parameters()):
                meta_param_dec.grad = meta_param_dec - inner_param_dec

        self.model_optimizer.step()

    def train(self):
        # meta-initialization
        self.run_epoch_reptile()
        # regular supervised learning
        self.run_epoch_supervise()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-Initialization")
    parser.add_argument("--learning_rate",
                                type=float,
                                help="learning rate",
                                default=1e-1)
    parser.add_argument("--inner_lr",
                                type=float,
                                help="inner learning rate",
                                default=1e-3)
    parser.add_argument("--inner_steps",
                                type=int,
                                help="inner_steps",
                                default=4)
    parser.add_argument("--sup_learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=5e-4)
    # ...... (Other options for your training recipe) 
    opts = parser.parse()

    trainer = Trainer(opts)
    trainer.train()

```

## <div align="">Citation</div>

    @inproceedings{wu2024boosting,
        title={Boosting Generalizability towards Zero-Shot Cross-Dataset Single-Image Indoor Depth by Meta-Initialization},
        author={Wu, Cho-Ying and Zhong, Yiqi and Wang, Junying and Neumann, Ulrich},
        booktitle={2024 IEEE/RSJ international conference on    intelligent robots and systems (IROS)},
        year={2024},
        organization={IEEE}
        }