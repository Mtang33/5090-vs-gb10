"""粘土关键帧 → 动画镜头，用 MiniMax H3（FL2VA int8 + 8 步 turbo LoRA，跑 6 步，和官方 i2v 模板一致）。
和 clay_anim.py 用同一种 shots.json；镜头可以多给一个 "last"（尾帧图，H3 支持首尾帧），或给 "refs"（参考图列表，走参考图生视频），也可以单独给 "length"。
H3 是 24 fps，帧数要在 17k+5 网格上（124 帧 ≈ 5.2 s）；只解码画面，不要音频。
用法（DGX）: [CLAY_ANIM=... CLAY_KEYS=...] python3 h3_anim.py shots.json <镜头 id ...> [--length 124] [--steps 6]"""
import json, sys, time, uuid, urllib.request, os, shutil
URL = 'http://127.0.0.1:8188'
ROOT = os.path.expanduser(os.environ.get('CLAY_ANIM', '~/clay_anim_h3')); os.makedirs(ROOT, exist_ok=True)
KEYS = os.path.expanduser(os.environ.get('CLAY_KEYS', '~/clay/keys'))
COMFY = os.path.expanduser('~/ComfyUI')


def wf_r2v(refs, prompt, w, h, length, seed, steps):
    """参考图生视频：ref2va 模型 + 4 步 turbo LoRA；提示词里用 <Picture 1>、<Picture 2> 指参考图。"""
    g = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_ref2va_int8_convrot.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors', 'strength_model': 1.0}},
        '3': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'type': 'minimax', 'device': 'default'}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}},
        '6': {'class_type': 'MiniMaxH3ReferenceToVideo', 'inputs': {'clip': ['3', 0], 'vae': ['4', 0], 'prompt': prompt, 'width': w, 'height': h,
                                                                    'length': length, 'ref_image_size': 'match'}},
        '7': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}},
        '8': {'class_type': 'BasicGuider', 'inputs': {'model': ['2', 0], 'conditioning': ['6', 0]}},
        '9': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'res_multistep'}},
        '10': {'class_type': 'BasicScheduler', 'inputs': {'model': ['2', 0], 'scheduler': 'simple', 'steps': steps, 'denoise': 1.0}},
        '11': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'noise': ['7', 0], 'guider': ['8', 0], 'sampler': ['9', 0], 'sigmas': ['10', 0], 'latent_image': ['6', 1]}},
        '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['4', 0]}},
        '13': {'class_type': 'SaveImage', 'inputs': {'images': ['12', 0], 'filename_prefix': 'h3anim/f'}},
    }
    for k, r in enumerate(refs):
        g[f'r{k}'] = {'class_type': 'LoadImage', 'inputs': {'image': r}}
        g['6']['inputs'][f'ref_images.ref_image_{k}'] = [f'r{k}', 0]
    return g


def wf(first, last, prompt, w, h, length, seed, steps):
    g = {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'minimax_h3_fl2va_int8_convrot.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors', 'strength_model': 1.0}},
        '3': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'type': 'minimax', 'device': 'default'}},
        '4': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'minimax_h3_video_vae_fp16.safetensors'}},
        '5': {'class_type': 'LoadImage', 'inputs': {'image': first}},
        '6': {'class_type': 'MiniMaxH3ImageToVideo', 'inputs': {'clip': ['3', 0], 'vae': ['4', 0], 'prompt': prompt, 'width': w, 'height': h,
                                                                 'length': length, 'first_frame': ['5', 0]}},
        '7': {'class_type': 'RandomNoise', 'inputs': {'noise_seed': seed}},
        '8': {'class_type': 'BasicGuider', 'inputs': {'model': ['2', 0], 'conditioning': ['6', 0]}},
        '9': {'class_type': 'KSamplerSelect', 'inputs': {'sampler_name': 'res_multistep'}},
        '10': {'class_type': 'BasicScheduler', 'inputs': {'model': ['2', 0], 'scheduler': 'simple', 'steps': steps, 'denoise': 1.0}},
        '11': {'class_type': 'SamplerCustomAdvanced', 'inputs': {'noise': ['7', 0], 'guider': ['8', 0], 'sampler': ['9', 0], 'sigmas': ['10', 0], 'latent_image': ['6', 1]}},
        '12': {'class_type': 'VAEDecode', 'inputs': {'samples': ['11', 0], 'vae': ['4', 0]}},
        '13': {'class_type': 'SaveImage', 'inputs': {'images': ['12', 0], 'filename_prefix': 'h3anim/f'}},
    }
    if last:
        g['14'] = {'class_type': 'LoadImage', 'inputs': {'image': last}}
        g['6']['inputs']['last_frame'] = ['14', 0]
    return g


def post(path, data):
    r = urllib.request.Request(URL + path, data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(r).read())


def get(path): return json.loads(urllib.request.urlopen(URL + path).read())


def run(prompt):
    pid = post('/prompt', {'prompt': prompt, 'client_id': str(uuid.uuid4())})['prompt_id']
    while True:
        h = get(f'/history/{pid}')
        if pid in h:
            st = h[pid].get('status', {})
            if st.get('status_str') == 'error': raise RuntimeError(json.dumps(st)[-1500:])
            imgs = [i for o in h[pid]['outputs'].values() for i in o.get('images', [])]
            if imgs: return imgs
        time.sleep(2.0)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opt = lambda k, d: int(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    length, steps = opt('--length', 124), opt('--steps', 6)
    args = [a for a in args if a not in (str(length), str(steps))]
    cfg = json.load(open(args[0])); shots = {s['id']: s for s in cfg['shots']}
    for sid in args[1:]:
        s = shots[sid]
        if s.get('refs'):                                   # 参考图生视频（R2V）
            refs = []
            for k, r in enumerate(s['refs']):
                n = f'h3r_{os.path.basename(ROOT)}_{sid}_{k}.png'; shutil.copy(f"{KEYS}/{r}", f'{COMFY}/input/{n}'); refs.append(n)
            t = time.time()
            imgs = run(wf_r2v(refs, f"{s['motion']} {cfg['style']}", cfg['w'], cfg['h'], s.get('length', length), s.get('seed', 42), s.get('steps', 4)))
        else:
            imgs = None
        src = None if imgs else (f"{ROOT}/{s['from_last']}/last.png" if s.get('from_last') else f"{KEYS}/{s['key']}")
        if src:
            first = f'h3_{os.path.basename(ROOT)}_{sid}.png'; shutil.copy(src, f'{COMFY}/input/{first}')
            last = None
            if s.get('last'):
                last = f'h3_{os.path.basename(ROOT)}_{sid}_last.png'; shutil.copy(f"{KEYS}/{s['last']}", f'{COMFY}/input/{last}')
            t = time.time()
            imgs = run(wf(first, last, f"{s['motion']} {cfg['style']}", cfg['w'], cfg['h'], s.get('length', length), s.get('seed', 42), steps))
        out = f'{ROOT}/{sid}'; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        for k, im in enumerate(imgs):
            shutil.copy(f"{COMFY}/output/{im['subfolder']}/{im['filename']}", f'{out}/{k:04d}.png')
            os.remove(f"{COMFY}/output/{im['subfolder']}/{im['filename']}")
        shutil.copy(f'{out}/{len(imgs) - 1:04d}.png', f'{out}/last.png')
        print(f'{sid}: {len(imgs)} 帧 {time.time() - t:.0f}s → {out}', flush=True)
