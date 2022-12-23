import numpy as np
from PIL import Image
from taming.models import vqgan
import sys
import torchvision, torch
from einops import rearrange
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List
from notebook_helpers import get_model
from ldm.util import ismap
from ldm.models.diffusion.ddim import DDIMSampler
import time

class SRData(Dataset):
    def __init__(self, dest, dest_out) -> None:
        super().__init__()
        self.dest = dest
        self.dest_out = dest_out
        assert os.path.exists(dest)
        self.file_list = [f for f in sorted(os.listdir(dest))]
        if os.path.exists(dest_out):
            for f in sorted(os.listdir(dest_out)):
                self.file_list.remove(f)
        else:
            os.makedirs(dest_out)

    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, index) -> Dict:
        filename = os.path.join(self.dest, self.file_list[index])
        example = dict()
        up_f = 4

        c = Image.open(filename)
        if c.mode != "RGB":
            c = c.convert("RGB")
        c = torchvision.transforms.ToTensor()(c)
        if c.shape != (3, 512, 512):
            print("Warning: Expected image shape (512, 512), get ", c.shape)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[1], up_f * c.shape[2]])
        c_up = rearrange(c_up, 'c h w -> h w c')
        c = rearrange(c, 'c h w -> h w c')
        c = 2. * c - 1.

        c = c.to(torch.device("cuda"))
        example["LR_image"] = c
        example["image"] = c_up
        example["out_dir"] = os.path.join(self.dest_out, self.file_list[index])
        return example

def run(model, example, task, custom_steps, resize_enabled=False, classifier_ckpt=None, global_step=None):

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = False
    temperature = 1.
    eta = 1.
    make_progrow = True
    custom_shape = None

    height, width = example["image"].shape[1:3]
    split_input = height >= 128 and width >= 128

    if split_input:
        ks = 128
        stride = 64
        vqf = 4  #
        model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
    else:
        if hasattr(model, "split_input_params"):
            delattr(model, "split_input_params")

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])

        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=False , masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):

    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,ddim_use_x0_pred=False):
    log = dict()

    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log

def save_images(batch, out_dir) -> None:
    batch = torch.clamp(batch, -1.0, 1.0)
    batch = (batch + 1.0) / 2.0 * 255
    batch = batch.detach().cpu()
    batch = batch.numpy().astype(np.uint8)
    batch = np.transpose(batch, (0, 2, 3, 1))
    for sample, dir in zip(batch, out_dir):
        Image.fromarray(sample).save(dir)

if __name__ == "__main__":
    argc = len(sys.argv)
    if argc <= 2:
        print("Usage: python infer.py model_log_dir config_name [checkpoint_name] [dirs to process]")
    else:
        model_log_dir = sys.argv[1]
        config_name = sys.argv[2]
        checkpoint_name = sys.argv[3] if argc >= 4 else "last.ckpt"

        dirs = sys.argv[4:] if argc >= 5 else os.listdir('./data/example_conditioning/superresolution')

    mode = "superresolution"

    log_dir = f"logs/{model_log_dir}"

    model = get_model(mode, os.path.join(log_dir, f"configs/{config_name}-project.yaml"), os.path.join(log_dir, f"checkpoints/{checkpoint_name}"))

    dest = f"data/example_conditioning/{mode}"

    for cur_dir in dirs:
        if cur_dir.endswith('output') or not os.path.isdir(os.path.join(dest, cur_dir)):
            continue

        print(f">>>>>>>Processing directory {cur_dir}>>>>>>>>")
        dataset = SRData(os.path.join(dest, cur_dir), os.path.join(dest, cur_dir+f'{model_log_dir}_{checkpoint_name}_output'))
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in tqdm(dataloader):
            custom_steps = 100
            logs = run(model['model'], batch, mode, custom_steps)
            img_lst = save_images(logs['sample'], batch['out_dir'])
        
        print(f"<<<<<<<<Finish Processing directory {cur_dir}<<<<<<")
    