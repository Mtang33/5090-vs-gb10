"""粘土关键帧 → 动画镜头：在 DGX 上用 ComfyUI API 跑 Wan 2.2 i2v 14B（fp8）+ lightx2v 4 步 LoRA（高噪/低噪两段采样）。
每个镜头：上传关键帧 → 生成 LENGTH 帧（16 fps）→ 存成 ~/clay_anim/<id>/%04d.png。
--from-last <id>：用另一个镜头的最后一帧当起始帧（长镜头接着往下演）。
用法（DGX）: [CLAY_ANIM=~/clay_anim/01 CLAY_KEYS=~/clay/clips/01/keys] python3 clay_anim.py shots.json <镜头 id ...>"""
import json, sys, time, uuid, urllib.request, os, glob, shutil
URL = 'http://127.0.0.1:8188'
ROOT = os.path.expanduser(os.environ.get('CLAY_ANIM', '~/clay_anim')); os.makedirs(ROOT, exist_ok=True)   # 每个片段用自己的目录
KEYS = os.path.expanduser(os.environ.get('CLAY_KEYS', '~/clay/keys'))
COMFY = os.path.expanduser('~/ComfyUI')


def wf(img_name, pos, neg, w, h, length, seed):
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors', 'weight_dtype': 'default'}},
        '3': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors', 'strength_model': 1.0}},
        '4': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['2', 0], 'lora_name': 'wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors', 'strength_model': 1.0}},
        '5': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['3', 0], 'shift': 5.0}},
        '6': {'class_type': 'ModelSamplingSD3', 'inputs': {'model': ['4', 0], 'shift': 5.0}},
        '7': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'umt5_xxl_fp16.safetensors', 'type': 'wan'}},
        '8': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'wan_2.1_vae.safetensors'}},
        '9': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['7', 0], 'text': pos}},
        '10': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['7', 0], 'text': neg}},
        '11': {'class_type': 'LoadImage', 'inputs': {'image': img_name}},
        '12': {'class_type': 'WanImageToVideo', 'inputs': {'positive': ['9', 0], 'negative': ['10', 0], 'vae': ['8', 0], 'start_image': ['11', 0],
                                                            'width': w, 'height': h, 'length': length, 'batch_size': 1}},
        '13': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['5', 0], 'add_noise': 'enable', 'noise_seed': seed, 'steps': 4, 'cfg': 1.0,
                                                             'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
                                                             'latent_image': ['12', 2], 'start_at_step': 0, 'end_at_step': 2, 'return_with_leftover_noise': 'enable'}},
        '14': {'class_type': 'KSamplerAdvanced', 'inputs': {'model': ['6', 0], 'add_noise': 'disable', 'noise_seed': seed, 'steps': 4, 'cfg': 1.0,
                                                             'sampler_name': 'euler', 'scheduler': 'simple', 'positive': ['12', 0], 'negative': ['12', 1],
                                                             'latent_image': ['13', 0], 'start_at_step': 2, 'end_at_step': 10000, 'return_with_leftover_noise': 'disable'}},
        '15': {'class_type': 'VAEDecode', 'inputs': {'samples': ['14', 0], 'vae': ['8', 0]}},
        '16': {'class_type': 'SaveImage', 'inputs': {'images': ['15', 0], 'filename_prefix': 'clayanim/f'}},
    }


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
    cfg = json.load(open(sys.argv[1])); want = sys.argv[2:]
    shots = {s['id']: s for s in cfg['shots']}
    for sid in want:
        s = shots[sid]
        src = f"{ROOT}/{s['from_last']}/last.png" if s.get('from_last') else f"{KEYS}/{s['key']}"
        name = f'clayanim_{os.path.basename(ROOT)}_{sid}.png'; shutil.copy(src, f'{COMFY}/input/{name}')
        pos = f"{s['motion']} {cfg['style']}"
        t = time.time()
        imgs = run(wf(name, pos, cfg['negative'], cfg['w'], cfg['h'], s.get('length', cfg['length']), s.get('seed', 42)))
        out = f'{ROOT}/{sid}'; shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        for k, im in enumerate(imgs):
            shutil.copy(f"{COMFY}/output/{im['subfolder']}/{im['filename']}", f'{out}/{k:04d}.png')
            os.remove(f"{COMFY}/output/{im['subfolder']}/{im['filename']}")
        shutil.copy(f'{out}/{len(imgs) - 1:04d}.png', f'{out}/last.png')
        print(f'{sid}: {len(imgs)} 帧 {time.time() - t:.0f}s → {out}', flush=True)
