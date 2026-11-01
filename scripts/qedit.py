"""Qwen-Image-Edit 2511（fp8，已合併 Lightning 4 步）修圖工作流，照 ComfyUI 官方 2511 模板：
參考圖縮到 1MP → TextEncodeQwenImageEditPlus（正負各一）→ index_timestep_zero → CFGNorm → 4 步 euler。
用不開 Sage 的 8189（和 Qwen-Image 一樣）。run() 沿用 clay_gen 的提交/輪詢。"""
import clay_gen


def wf(image, prompt, seed, steps=4):
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_edit_2511_fp8_e4m3fn_scaled_lightning_comfyui_4steps_v1.0.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'type': 'qwen_image'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        '4': {'class_type': 'LoadImage', 'inputs': {'image': image}},
        '5': {'class_type': 'ImageScaleToTotalPixels', 'inputs': {'image': ['4', 0], 'upscale_method': 'lanczos', 'megapixels': 1.0, 'resolution_steps': 8}},
        '6': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['2', 0], 'vae': ['3', 0], 'image1': ['5', 0], 'prompt': prompt}},
        '7': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {'clip': ['2', 0], 'vae': ['3', 0], 'image1': ['5', 0], 'prompt': ''}},
        '8': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {'conditioning': ['6', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '9': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {'conditioning': ['7', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '10': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['5', 0], 'vae': ['3', 0]}},
        '11': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['1', 0], 'shift': 3.1}},
        '12': {'class_type': 'CFGNorm', 'inputs': {'model': ['11', 0], 'strength': 1.0}},
        '13': {'class_type': 'KSampler', 'inputs': {'model': ['12', 0], 'positive': ['8', 0], 'negative': ['9', 0], 'latent_image': ['10', 0],
                                                     'seed': seed, 'steps': steps, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '14': {'class_type': 'VAEDecode', 'inputs': {'samples': ['13', 0], 'vae': ['3', 0]}},
        '15': {'class_type': 'SaveImage', 'inputs': {'images': ['14', 0], 'filename_prefix': 'qedit'}},
    }


def run(prompt): return clay_gen.run(prompt)
