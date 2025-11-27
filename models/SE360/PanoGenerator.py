import os
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, OmniGenTransformer2DModel, OmniGenPipeline
from diffusers.image_processor import VaeImageProcessor
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
from utils.pano import pad_pano, unpad_pano
from ..modules.utils import WandbLightningModule
from typing import Optional
from .processor_omnigen import OmniGenMultiModalProcessor


class PanoBase(WandbLightningModule):
    def __init__(
            self,
            pano_prompt_prefix: str = '',
            pers_prompt_prefix: str = '',
            mv_pano_prompt: bool = False,
            copy_pano_prompt: bool = False,
            ):
        super().__init__()
        self.save_hyperparameters()

    def add_pano_prompt_prefix(self, pano_prompt):
        if isinstance(pano_prompt, str):
            if pano_prompt == '':
                return ''
            if self.hparams.pano_prompt_prefix == '':
                return pano_prompt
            return ' '.join([self.hparams.pano_prompt_prefix, pano_prompt])
        return [self.add_pano_prompt_prefix(p) for p in pano_prompt]

    def add_pers_prompt_prefix(self, pers_prompt):
        if isinstance(pers_prompt, str):
            if pers_prompt == '':
                return ''
            if self.hparams.pers_prompt_prefix == '':
                return pers_prompt
            return ' '.join([self.hparams.pers_prompt_prefix, pers_prompt])
        return [self.add_pers_prompt_prefix(p) for p in pers_prompt]

    # def get_pano_prompt(self, batch):
    #     pano_prompt = batch['pano_prompt']
    #     return pano_prompt
    def get_pano_prompt(self, batch):
        if self.hparams.mv_pano_prompt:
            prompts = list(map(list, zip(*batch['prompt'])))
            pano_prompt = ['. '.join(p1) if p2 else '' for p1, p2 in zip(prompts, batch['pano_prompt'])]
        else:
            pano_prompt = batch['pano_prompt']
        return self.add_pano_prompt_prefix(pano_prompt)
    def get_forward_prompt(self, batch):
        forward_prompt = batch['forward_prompt']
        return forward_prompt
    def get_reverse_prompt(self, batch):
        reverse_prompt = batch['reverse_prompt']
        return reverse_prompt
    def get_edited_prompt(self, batch):
        edited_prompt = batch['edited_prompt']
        return edited_prompt
    

    
    def get_pers_prompt(self, batch):
        if self.hparams.copy_pano_prompt:
            prompts = sum([[p] * batch['cameras']['height'].shape[-1] for p in batch['pano_prompt']], [])# Create a new list, repeat this prompt height.shape[-1] times, sum(..., []) flattens the generated nested list into a one-dimensional list
        else:
            prompts = sum(map(list, zip(*batch['prompt'])), [])
        return self.add_pers_prompt_prefix(prompts)


class PanoGenerator(PanoBase):
    def __init__(
            self,
            lr: float = 3e-4,
            guidance_scale: float = 2.5,
            image_guidance_scale: float = 1.6,
            model_id: Optional[str] = 'Shitao/OmniGen-v1-diffusers',
            inference_timesteps: int = 50,  # Number of timesteps during inference
            image_use_prob: float = 0.99, # 0.01 for unconditioned input
            ref_use_prob: float = 0, # Reference image usage probability
            edit_mask_use_prob: float = 0.2, # Edit mask usage probability
            latent_pad: int = 8,
            pano_lora: bool = True,
            train_pano_lora: bool = True,
            pers_lora: bool = True,
            train_pers_lora: bool = True,
            lora_rank: int = 16,
            ckpt_path: Optional[str] = None,
            rot_diff: float = 90.0,
            unet_pad: bool = True,
            use_cube: bool = False,
            use_gradient_checkpointing: bool = False,
            use_mask_in_inference: bool = False,
            use_ref_in_inference: bool = False,
            huggingface_cache: Optional[str] = None, 
            **kwargs
            ):
        super().__init__(**kwargs)
        self.trainable_params = []
        self.save_hyperparameters()
        if ckpt_path is not None:
            self.hparams.ckpt_path = ckpt_path
        self.load_shared()
        # self.instantiate_model()
        # if ckpt_path is not None:
        #     print(f"Loading weights from {ckpt_path}")
        #     state_dict = torch.load(ckpt_path, weights_only=True)['state_dict']
        #     self.convert_state_dict(state_dict)
        #     # try:
        #     self.load_state_dict(state_dict, strict=True)
            # except RuntimeError as e:
            #     print(e)
            #     self.load_state_dict(state_dict, strict=False)

    def exclude_eval_metrics(self, checkpoint):
        for key in list(checkpoint['state_dict'].keys()):
            if key.startswith('eval_metrics'):
                del checkpoint['state_dict'][key]

    def convert_state_dict(self, state_dict):

        # Count LoRA-related keys
        lora_keys_count = 0
        for key in state_dict.keys():
            if 'mv_base_model.omnigen_transformer._orig_mod' in key and ('lora_A' in key or 'lora_B' in key):
                lora_keys_count += 1
        
        print(f"Detected {lora_keys_count} LoRA-related keys in weight file, format is correct, no conversion needed")

    def on_load_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)
        self.convert_state_dict(checkpoint['state_dict'])

    def on_save_checkpoint(self, checkpoint):
        self.exclude_eval_metrics(checkpoint)

    def load_shared(self):#"hf-internal-testing/llama-tokenizer"
        # self.tokenizer = LlamaTokenizer.from_pretrained(self.hparams.model_id, use_safetensors=True,cache_dir=self.hparams.huggingface_cache)
        # self.tokenizer.requires_grad_(False)
        self.tokenizer = OmniGenPipeline.from_pretrained(self.hparams.model_id, use_safetensors=True,cache_dir=self.hparams.huggingface_cache).tokenizer
        self.vae = AutoencoderKL.from_pretrained(
        self.hparams.model_id, subfolder="vae", torch_dtype=torch.float32, use_safetensors=True,cache_dir=self.hparams.huggingface_cache)
        self.vae.eval()
        self.vae.requires_grad_(False)
        # Disable VAE compilation to support attention visualization
        self.vae = torch.compile(self.vae)

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.multimodal_processor = OmniGenMultiModalProcessor(self.tokenizer, max_image_size=1024)
        self.tokenizer_max_length = 120000
        # (
        #     self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 120000
        # )
        self.default_sample_size = 128
        # Load scheduler
        
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.hparams.model_id,
            subfolder="scheduler",
            torch_dtype=torch.float32,
            use_safetensors=True,
            cache_dir=self.hparams.huggingface_cache
        )

    def add_lora(self, omnigen):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=16,  
            lora_alpha=16,  
            target_modules=[
                "to_q",         
                "to_k",         
                "to_v",         
                "to_out.0",             
            ],
            lora_dropout=0., 
            bias="none",       
        )
        omnigen = get_peft_model(omnigen, lora_config)
        print("add LoRA adapter completed.")

        lora_trainable_params = [p for p in omnigen.parameters() if p.requires_grad]
        return (lora_trainable_params, 1.0)



    def load_branch(self, add_lora, train_lora):
        omnigen = OmniGenTransformer2DModel.from_pretrained(
            self.hparams.model_id, subfolder="transformer", torch_dtype=torch.float32, use_safetensors=True, cache_dir=self.hparams.huggingface_cache
        )
        # from xformers.ops import MemoryEfficientAttentionFlashAttentionOp  
        # omnigen.enable_xformers_memory_efficient_attention()
        self.transformer_channel = omnigen.config.in_channels
        if self.hparams.use_gradient_checkpointing: 
            omnigen.enable_gradient_checkpointing() 
        # omnigen.requires_grad_(False)

        # print(omnigen) 
        if hasattr(omnigen, "add_adapter"):
            print("model supports add_adapter method, using it to add LoRA...")

        if add_lora:
            params = self.add_lora(omnigen)
            if train_lora:
                self.trainable_params.append(params)
        # disable compilation to support attention visualization
        print("Compiling OmniGen...")
        omnigen = torch.compile(omnigen)
        print("Omnigen compiled.")


        return omnigen

    def load_pano(self):
        return self.load_branch(
            self.hparams.pano_lora,
            self.hparams.train_pano_lora,
        )

    def load_pers(self):
        return self.load_branch(
            self.hparams.pers_lora,
            self.hparams.train_pers_lora,
        )


    @torch.no_grad()
    def encode_text(self, text):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids#(20,77), each element is an integer, representing the index of the corresponding word or subword in the model vocabulary
 
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device), attention_mask=attention_mask)

        return prompt_embeds[0].to(self.dtype)


    @torch.no_grad()
    def encode_image(
        self,
        input_pixel_values,
        device: torch.device,
    ):
        """
        get the continue embedding of input images by VAE

        Args:
            input_pixel_values: normlized pixel of input images
            device:
        Returns: torch.Tensor
        """
        device = device 
        dtype = self.vae.dtype

        input_img_latents = []
        for img in input_pixel_values:
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            img = self.vae.encode(img.to(device, dtype)).latent_dist.sample().mul_(self.vae.config.scaling_factor)
            input_img_latents.append(img)
        return input_img_latents

    def check_inputs(
        self,
        prompt,
        input_images,
        height,
        width,
        use_input_image_size_as_output,
        callback_on_step_end_tensor_inputs=None,
    ):
        if input_images is not None:
            if len(input_images) != len(prompt):
                raise ValueError(
                    f"The number of prompts: {len(prompt)} does not match the number of input images: {len(input_images)}."
                )
            # for i in range(len(input_images)):
            #     if input_images[i] is not None:
            #         if not all(f"<img><|image_{k + 1}|></img>" in prompt[i] for k in range(len(input_images[i]))):
            #             raise ValueError(
            #                 f"prompt `{prompt[i]}` doesn't have enough placeholders for the input images `{input_images[i]}`"
            #             )

        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            print(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if use_input_image_size_as_output:
            if input_images is None or input_images[0] is None:
                raise ValueError(
                    "`use_input_image_size_as_output` is set to True, but no input image was found. If you are performing a text-to-image task, please set it to False."
                )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
            
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()
        
    def pad_pano(self, pano, latent=False):
        b, m = pano.shape[:2]
        padding = self.hparams.latent_pad
        if not latent:
            padding *= 8
        return pad_pano(pano, padding=padding)

    def unpad_pano(self, pano_pad, latent=False):
        padding = self.hparams.latent_pad
        if not latent:
            padding *= 8
        return unpad_pano(pano_pad, padding=padding)

    def gen_cls_free_guide_pair(self, *inputs):
        result = []
        for input in inputs:
            if input is None:
                result.append(None)
            elif isinstance(input, dict):
                result.append({k: torch.cat([v]*3) for k, v in input.items()})
            elif isinstance(input, list):
                result.append([torch.cat([v]*3) for v in input])
            else:
                result.append(torch.cat([input]*3))
        return result

    def combine_cls_free_guide_pred(self, *noise_pred_list):
        result = []
        for noise_pred in noise_pred_list:
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
            noise_pred = (
                noise_pred_uncond
                + self.hparams.guidance_scale * (noise_pred_text - noise_pred_image)
                + self.hparams.image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )
            result.append(noise_pred)
        if len(result) == 1:
            return result[0]
        return result

    def rotate_latent(self, pano_latent, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent
        return torch.roll(pano_latent, int(degree / 360 * pano_latent.shape[-1]), dims=-1)

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b = latents.shape[0]
        latents = (1 / vae.config.scaling_factor * latents)
        latents = rearrange(latents, 'b m c h w -> (b m) c h w')
        # No gradients needed during inference, decode directly
        image = vae.decode(latents.to(vae.dtype)).sample
        image = rearrange(image, '(b m) c h w -> b m c h w', b=b)
        return image.to(self.dtype)

    def configure_optimizers(self):
        param_groups = []
        for params, lr_scale in self.trainable_params:
            # Filter parameters where requires_grad is True
            param_groups.append({"params": params, "lr": self.hparams.lr * lr_scale})
        
        if not param_groups:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Warning: No trainable parameters found for the optimizer.!!!!!!!!!!!!!!!!!!!!!!!!!")
            # return None 
            pass

        optimizer = torch.optim.AdamW(param_groups)

        # scheduler = {
        #     'scheduler': CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-7),
        #     'interval': 'epoch',  # update the learning rate after each epoch
        #     'name': 'cosine_annealing_lr',
        # }
        scheduler = {
            'scheduler': CosineAnnealingLR(optimizer, T_max=self.trainer.estimated_stepping_batches, eta_min=1e-7), # Use estimated_stepping_batches as T_max
            'interval': 'step', 
            'name': 'cosine_annealing_lr',
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        output_dir = os.path.join(self.logger.save_dir, 'test', batch['pano_id'][0])
        self.inference_and_save(batch, output_dir)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dir = os.path.join(self.logger.save_dir, 'predict', batch['pano_id'][0])
        self.inference_and_save(batch, output_dir, 'jpg')
