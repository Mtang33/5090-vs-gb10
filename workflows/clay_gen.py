"""粘土定帧风格关键帧：在 DGX 上用 ComfyUI API 跑 Qwen-Image 2512（fp8）+ Lightning 4 步 LoRA。
在 DGX 上运行（ComfyUI 监听 127.0.0.1:8188），读 clay_scenes.json，每个场景出 N 张候选到 ~/clay_out/<场景>_<k>.png。
用法（DGX）: [CLAY_OUT=~/clay_out/01] python3 clay_gen.py scenes.json [场景 id ...] [--n 2] [--steps 4]"""
import json, sys, time, uuid, urllib.request, os, random
URL = os.environ.get('CLAY_URL', 'http://127.0.0.1:8189')   # Qwen-Image 用不带 SageAttention 的那个 ComfyUI（开了 Sage 会出全黑图）
OUT = os.path.expanduser(os.environ.get('CLAY_OUT', '~/clay_out')); os.makedirs(OUT, exist_ok=True)   # 每个片段用自己的目录


def wf(pos, neg, seed, w, h, steps):
    return {
        '1': {'class_type': 'UNETLoader', 'inputs': {'unet_name': 'qwen_image_2512_fp8_e4m3fn.safetensors', 'weight_dtype': 'default'}},
        '2': {'class_type': 'CLIPLoader', 'inputs': {'clip_name': 'qwen_2.5_vl_7b_fp8_scaled.safetensors', 'type': 'qwen_image'}},
        '3': {'class_type': 'VAELoader', 'inputs': {'vae_name': 'qwen_image_vae.safetensors'}},
        '4': {'class_type': 'LoraLoaderModelOnly', 'inputs': {'model': ['1', 0], 'lora_name': 'Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors', 'strength_model': 1.0}},
        '5': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['4', 0], 'shift': 3.1}},
        '6': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': pos}},
        '7': {'class_type': 'CLIPTextEncode', 'inputs': {'clip': ['2', 0], 'text': neg}},
        '8': {'class_type': 'EmptySD3LatentImage', 'inputs': {'width': w, 'height': h, 'batch_size': 1}},
        '9': {'class_type': 'KSampler', 'inputs': {'model': ['5', 0], 'positive': ['6', 0], 'negative': ['7', 0], 'latent_image': ['8', 0],
                                                    'seed': seed, 'steps': steps, 'cfg': 1.0, 'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '10': {'class_type': 'VAEDecode', 'inputs': {'samples': ['9', 0], 'vae': ['3', 0]}},
        '11': {'class_type': 'SaveImage', 'inputs': {'images': ['10', 0], 'filename_prefix': 'clay'}},
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
            if st.get('status_str') == 'error': raise RuntimeError(json.dumps(st)[:800])
            imgs = [i for o in h[pid]['outputs'].values() for i in o.get('images', [])]
            if imgs: return imgs[0]
        time.sleep(1.0)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    n = int(sys.argv[sys.argv.index('--n') + 1]) if '--n' in sys.argv else 2
    steps = int(sys.argv[sys.argv.index('--steps') + 1]) if '--steps' in sys.argv else 4
    if '--n' in sys.argv: args.remove(str(n))
    if '--steps' in sys.argv: args.remove(str(steps))
    cfg = json.load(open(args[0])); want = set(args[1:])
    for sc in cfg['scenes']:
        if want and sc['id'] not in want: continue
        pos = f"{sc['prompt']} {cfg['style']}"
        for k in range(n):
            seed = sc.get('seed', 1000) + k * 7919
            t = time.time(); img = run(wf(pos, sc.get('negative') or cfg['negative'], seed, cfg['w'], cfg['h'], steps))   # 场景可以有自己的负面提示
            q = f"filename={img['filename']}&subfolder={img['subfolder']}&type={img['type']}"
            data = urllib.request.urlopen(f'{URL}/view?{q}').read()
            p = f"{OUT}/{sc['id']}_{k}.png"; open(p, 'wb').write(data)
            print(f"{sc['id']}_{k}  seed {seed}  {time.time() - t:.1f}s  → {p}", flush=True)
