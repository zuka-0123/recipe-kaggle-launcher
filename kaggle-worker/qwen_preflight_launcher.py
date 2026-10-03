"""One-off free T4 load/context check; no recipe, Cloudflare or Drive access."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from launcher import generic_bundle, kernel_status, public_repository, LaunchError, status_failure_code

def main():
    public_repository(os.environ.get('GITHUB_REPOSITORY'), os.environ.get('GITHUB_TOKEN', ''))
    owner = os.environ.get('KAGGLE_KERNEL_OWNER', '')
    slug = os.environ.get('KAGGLE_KERNEL_SLUG', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', owner) or not re.fullmatch(r'[a-z0-9-]+', slug):
        raise LaunchError('invalid_kernel_ref')
    # The usual worker must be terminal, even though this check has no D1 jobs.
    kernel_status(owner + '/' + slug)
    reference = owner + '/recipe-qwen-awq-preflight'
    # This dedicated preflight now exists; reject an active version too.
    kernel_status(reference)
    code = '''import base64, gc, io, json, os, pathlib, shutil, subprocess, sys, tempfile, time, zipfile
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
root = pathlib.Path(tempfile.mkdtemp(prefix='recipe-awq-preflight-', dir='/tmp'))
stage = 'install'
try:
    zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(root)
    sys.path.insert(0, str(root))
    installed = subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q', '-r', str(root / 'requirements-worker.txt')], capture_output=True, text=True, timeout=600)
    if installed.returncode:
        raise RuntimeError('dependency_install_failed')
    import torch
    from worker import Models, t4_efficient_sdpa, awq_gpu_dequant, safe_job_diagnostic
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7,5):
        raise RuntimeError('free_t4_required')
    model = Models({'llm_model':'Qwen/Qwen3-14B-AWQ'}, {}, '', root, time.time()+3600)
    stage = 'load'
    started = time.monotonic()
    model.load('llm')
    print('recipe_preflight stage=load result=ok seconds=' + str(round(time.monotonic()-started,1)), flush=True)
    tokenizer = model.processor
    prompt = 'Return exactly this JSON object without explanation: {"ok":true,"unknown":null}'
    for long_context in [False, True]:
        stage = 'generate'
        context = ('This diagnostic document contains no recipe facts. Unknown values remain null.\\n' * 700) if long_context else ''
        messages = [{'role':'system','content':'Return only the requested JSON. Do not use thinking.'}, {'role':'user','content':context + prompt}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs = tokenizer([text],return_tensors='pt').to('cuda:0')
        count = inputs['input_ids'].shape[-1]
        if count > 12000:
            raise RuntimeError('preflight_context_too_large')
        torch.cuda.reset_peak_memory_stats(0)
        torch.manual_seed(42)
        started = time.monotonic()
        with torch.inference_mode(), t4_efficient_sdpa(), awq_gpu_dequant():
            output = model.model.generate(**inputs,max_new_tokens=128,do_sample=True,temperature=0.7,top_p=0.8,top_k=20,min_p=0.0,max_time=300)
        raw = tokenizer.decode(output[0,count:],skip_special_tokens=True)
        valid = json.loads(raw) == {'ok':True,'unknown':None} and '<think>' not in raw
        print('recipe_preflight stage=generate input_tokens=' + str(count) + ' output_tokens=' + str(output.shape[-1]-count) + ' seconds=' + str(round(time.monotonic()-started,1)) + ' valid_json=' + str(valid).lower() + ' allocated_mib=' + str(round(torch.cuda.memory_allocated(0)/1048576)) + ' reserved_mib=' + str(round(torch.cuda.memory_reserved(0)/1048576)) + ' peak_mib=' + str(round(torch.cuda.max_memory_allocated(0)/1048576)) + ' total_mib=' + str(round(torch.cuda.get_device_properties(0).total_memory/1048576)),flush=True)
        del inputs, output
        gc.collect(); torch.cuda.empty_cache()
    model.unload()
except Exception as error:
    if 'safe_job_diagnostic' in locals():
        safe_job_diagnostic('structure' if stage == 'generate' else 'extract',error)
    print('recipe_preflight stage=' + stage + ' result=failed exception=' + type(error).__name__,flush=True)
finally:
    shutil.rmtree(root,ignore_errors=True)
'''
    code = 'BUNDLE = ' + repr(generic_bundle()) + '\n' + code
    notebook = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'id':'free-t4-check','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)}]}
    with tempfile.TemporaryDirectory(prefix='recipe-preflight-launch-') as temporary:
        directory = Path(temporary)
        (directory/'worker.ipynb').write_text(json.dumps(notebook),encoding='utf-8')
        metadata={'id':reference,'title':'Recipe Qwen AWQ preflight','code_file':'worker.ipynb','language':'python','kernel_type':'notebook','is_private':True,'enable_gpu':True,'enable_internet':True,'machine_shape':'NvidiaTeslaT4','dataset_sources':[],'competition_sources':[],'kernel_sources':[]}
        (directory/'kernel-metadata.json').write_text(json.dumps(metadata),encoding='utf-8')
        result=subprocess.run(['kaggle','kernels','push','-p',str(directory),'--accelerator','NvidiaTeslaT4','--timeout','3600'],capture_output=True,text=True,timeout=180)
        if result.returncode or not re.search(r'Kernel version(?: \d+)? successfully pushed\.',result.stdout or ''):
            raise LaunchError(status_failure_code((result.stdout or '')+'\n'+(result.stderr or '')))
    print('Private free T4 preflight submitted. No recipe or Cloudflare access.')

if __name__ == '__main__':
    try:
        main()
    except LaunchError as error:
        print('Preflight stopped: '+error.code)
        raise SystemExit(1) from None
    except Exception:
        print('Preflight stopped: unknown_error')
        raise SystemExit(1) from None
