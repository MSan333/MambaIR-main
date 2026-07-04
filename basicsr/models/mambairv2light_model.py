import os
import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class MambaIRv2LightModel(SRModel):
    """MambaIRv2 lightSR model for image restoration.

    只保存 latest 和 best 模型，不保存每次 checkpoint 的 per-iter 副本。
    """

    def save(self, epoch, current_iter):
        """Save only latest model and training state (overwrite, no per-iter copies)."""
        # 始终覆盖保存为 net_g_latest.pth
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', -1,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', -1)
        # 保存训练状态为 latest.state（用于 auto-resume，覆盖）
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter,
                     'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_path = os.path.join(self.opt['path']['training_states'],
                                     'latest.state')
            torch.save(state, save_path)

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        """Update best metric and save best model when strictly improved."""
        # 判断是否严格优于当前最佳
        is_better = False
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val > self.best_metric_results[dataset_name][metric]['val']:
                is_better = True
        else:
            if val < self.best_metric_results[dataset_name][metric]['val']:
                is_better = True

        # 调用父类更新最佳记录
        super()._update_best_metric_result(dataset_name, metric, val,
                                            current_iter)

        # 指标提升时保存 best 模型
        if is_better:
            logger = get_root_logger()
            logger.info(f'New best {metric}: {val:.4f} @ {current_iter} iter. '
                        f'Saving best model.')
            if hasattr(self, 'net_g_ema'):
                self.save_network([self.net_g, self.net_g_ema], 'net_g', 'best',
                                  param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g, 'net_g', 'best')

    # test by partitioning
    def test(self):
        _, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1  # number of horizontal cut sections
        split_token_w = w // 200 + 1  # number of vertical cut sections
        # padding
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, H, W = img.size()
        split_h = H // split_token_h  # height of each partition
        split_w = W // split_token_w  # width of each partition
        # overlapping
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get('scale', 1)
        ral = H // split_h
        row = W // split_w
        slices = []  # list of partition borders
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i*split_h, (i+1)*split_h+shave_h)
                elif i == ral - 1:
                    top = slice(i*split_h-shave_h, (i+1)*split_h)
                else:
                    top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j*split_w, (j+1)*split_w)
                elif j == 0:
                    left = slice(j*split_w, (j+1)*split_w+shave_w)
                elif j == row - 1:
                    left = slice(j*split_w-shave_w, (j+1)*split_w)
                else:
                    left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                temp = (top, left)
                slices.append(temp)
        img_chops = []  # list of partitions
        for temp in slices:
            top, left = temp
            img_chops.append(img[..., top, left])
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g_ema(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = []
                for chop in img_chops:
                    out = self.net_g(chop)  # image processing of each partition
                    outputs.append(out)
                _img = torch.zeros(1, C, H * scale, W * scale)
                # merge
                for i in range(ral):
                    for j in range(row):
                        top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h * scale, (shave_h + split_h) * scale)
                        if j == 0:
                            _left = slice(0, split_w * scale)
                        else:
                            _left = slice(shave_w * scale, (shave_w + split_w) * scale)
                        _img[..., top, left] = outputs[i * row + j][..., _top, _left]
                self.output = _img
            self.net_g.train()
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
