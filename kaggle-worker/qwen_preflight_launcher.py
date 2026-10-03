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
    from contextlib import contextmanager, redirect_stdout, redirect_stderr
    import importlib
    from worker import Models, t4_efficient_sdpa, ensure_awq_import_compat, safe_job_diagnostic
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7,5):
        raise RuntimeError('free_t4_required')
    stage = 'kernel_check'
    ensure_awq_import_compat()
    gemm = importlib.import_module('awq.modules.linear.gemm')
    if gemm.TRITON_AVAILABLE is not True or not callable(getattr(gemm,'awq_gemm_triton',None)) or not callable(getattr(gemm,'awq_dequantize_triton',None)):
        raise RuntimeError('triton_awq_unavailable')
    from awq.utils.packing_utils import dequantize_gemm
    @contextmanager
    def preflight_triton_awq():
        original = (gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned)
        try:
            gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned = None, True, True
            # Compiler failures may print paths or code. Keep them out of Notebook logs.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                yield
        finally:
            gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned = original
    torch.manual_seed(42)
    qweight = torch.randint(-(2**31),2**31,(512,32),device='cuda:0',dtype=torch.int32)
    qzeros = torch.randint(-(2**31),2**31,(4,32),device='cuda:0',dtype=torch.int32)
    scales = (torch.rand((4,256),device='cuda:0',dtype=torch.float16)+0.5)*0.05
    reference = dequantize_gemm(qweight,qzeros,scales,4,128)
    started = time.monotonic()
    with torch.inference_mode(), preflight_triton_awq():
        restored = gemm.awq_dequantize_triton(qweight,scales,qzeros)
        torch.testing.assert_close(restored,reference,rtol=0.001,atol=0.001)
        maximum_error = 0.0
        for rows in [1,16]:
            matrix = torch.randn((rows,512),device='cuda:0',dtype=torch.float16)
            result = gemm.awq_gemm_triton(matrix,qweight,scales,qzeros,split_k_iters=8)
            expected = torch.matmul(matrix,reference)
            torch.testing.assert_close(result,expected,rtol=0.02,atol=0.1)
            maximum_error = max(maximum_error,float((result-expected).abs().max().item()))
        torch.cuda.synchronize(0)
    print('recipe_preflight stage=kernel_check result=ok seconds=' + str(round(time.monotonic()-started,3)) + ' max_abs_error=' + str(round(maximum_error,6)),flush=True)
    del qweight,qzeros,scales,reference,restored,matrix,result,expected
    gc.collect(); torch.cuda.empty_cache()
    model = Models({'llm_model':'Qwen/Qwen3-14B-AWQ'}, {}, '', root, time.time()+3600)
    stage = 'load'
    started = time.monotonic()
    model.load('llm')
    print('recipe_preflight stage=load result=ok seconds=' + str(round(time.monotonic()-started,1)), flush=True)
    tokenizer = model.processor
    prompt = 'Return exactly this JSON object without explanation: {"ok":true,"unknown":null}'
    for check in ['short_json','decode_128','long_json']:
        stage = 'decode' if check == 'decode_128' else 'generate'
        context = ('This diagnostic document contains no recipe facts. Unknown values remain null.\\n' * 700) if check == 'long_json' else ''
        request = 'Continue writing the word sample separated by spaces. Do not stop early. No explanations.' if check == 'decode_128' else context + prompt
        system = 'Do not use thinking. Follow the requested output format.' if check == 'decode_128' else 'Return only the requested JSON. Do not use thinking.'
        messages = [{'role':'system','content':system}, {'role':'user','content':request}]
        text = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs = tokenizer([text],return_tensors='pt').to('cuda:0')
        count = inputs['input_ids'].shape[-1]
        if count > 12000:
            raise RuntimeError('preflight_context_too_large')
        torch.cuda.reset_peak_memory_stats(0)
        torch.manual_seed(42)
        started = time.monotonic()
        options = {'min_new_tokens':128} if check == 'decode_128' else {}
        with torch.inference_mode(), t4_efficient_sdpa(), preflight_triton_awq():
            output = model.model.generate(**inputs,max_new_tokens=128,do_sample=True,temperature=0.7,top_p=0.8,top_k=20,min_p=0.0,max_time=600 if check == 'decode_128' else 300,**options)
        torch.cuda.synchronize(0)
        elapsed = time.monotonic()-started
        raw = tokenizer.decode(output[0,count:],skip_special_tokens=True)
        valid = (output.shape[-1]-count == 128) if check == 'decode_128' else json.loads(raw) == {'ok':True,'unknown':None}
        valid = valid and '<think>' not in raw and '</think>' not in raw
        print('recipe_preflight stage=' + stage + ' input_tokens=' + str(count) + ' output_tokens=' + str(output.shape[-1]-count) + ' seconds=' + str(round(elapsed,1)) + ' tokens_per_second=' + str(round((output.shape[-1]-count)/elapsed,3)) + ' valid=' + str(valid).lower() + ' allocated_mib=' + str(round(torch.cuda.memory_allocated(0)/1048576)) + ' reserved_mib=' + str(round(torch.cuda.memory_reserved(0)/1048576)) + ' peak_mib=' + str(round(torch.cuda.max_memory_allocated(0)/1048576)) + ' total_mib=' + str(round(torch.cuda.get_device_properties(0).total_memory/1048576)),flush=True)
        if not valid:
            raise RuntimeError('preflight_output_invalid')
        del inputs, output
        gc.collect(); torch.cuda.empty_cache()
    model.unload()
except Exception as error:
    if 'safe_job_diagnostic' in locals():
        safe_job_diagnostic('structure' if stage in ['generate','decode'] else 'extract',error)
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
