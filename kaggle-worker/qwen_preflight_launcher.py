"""One-off native AWQ free T4 check; no recipe, Cloudflare or Drive access."""
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
    code = '''import base64, gc, importlib.metadata, io, json, os, pathlib, re, shutil, signal, subprocess, sys, sysconfig, tempfile, time, urllib.request, zipfile
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
root = pathlib.Path(tempfile.mkdtemp(prefix='recipe-awq-preflight-', dir='/tmp'))
stage = 'install'
def run_captured(command, timeout, cwd=None, env=None):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # pip/ninja/nvcc children share this process group; stop all of them.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise RuntimeError('bounded_command_timeout') from None
    if process.returncode:
        raise RuntimeError('bounded_command_failed')
    return stdout

def select_awq_extension(source):
    marker = 'additional_setup_kwargs = {'
    if source.count(marker) != 1:
        raise RuntimeError('native_setup_shape_unknown')
    # Keep the official awq_ext source/flags, omit unused Ampere/exllama modules.
    return source.replace(marker,
        'extensions = [extension for extension in extensions if extension.name == "awq_ext"]\\n' + marker)

def toolkit_release(output, torch_cuda):
    match = re.search(r'release (\\d{1,2})\\.(\\d{1,2})(?!\\d)', output) if isinstance(output,str) else None
    expected = re.fullmatch(r'(\\d{1,2})\\.(\\d{1,2})', torch_cuda) if isinstance(torch_cuda,str) else None
    if not match or not expected or int(match.group(1)) != int(expected.group(1)):
        raise RuntimeError('native_toolkit_mismatch')
    # Match Torch's official extension check: a CUDA minor difference is a warning.
    # Do not bypass its later compiler/version checks or spoof any version values.
    if int(match.group(2)) != int(expected.group(2)):
        print('recipe_preflight stage=toolkit_check warning_cuda_minor_mismatch=1',flush=True)
    return tuple(map(int, match.groups()))

def safe_toolkit_diagnostic(output, torch_cuda):
    fields = []
    for label,value,pattern in [('nvcc',output,r'release (\\d{1,2})\\.(\\d{1,2})(?!\\d)'),
            ('torch_cuda',torch_cuda,r'^(\\d{1,2})\\.(\\d{1,2})$')]:
        match = re.search(pattern,value) if isinstance(value,str) else None
        if match:
            major,minor = map(int,match.groups())
            fields.extend([label + '_major=' + str(major),label + '_minor=' + str(minor)])
    if fields:
        print('recipe_preflight stage=toolkit_check ' + ' '.join(fields),flush=True)

def safe_failure_reason(error):
    allowed = {'bounded_command_timeout','bounded_command_failed','native_setup_shape_unknown',
        'native_toolkit_mismatch','torch_changed','free_t4_required','native_toolkit_missing',
        'python_headers_missing','native_compiler_missing','native_compiler_incompatible',
        'native_source_too_large','native_source_path_invalid','native_build_invalid',
        'native_awq_unavailable','preflight_context_too_large','preflight_output_invalid'}
    if type(error) is RuntimeError:
        value = str(error)
        return value if value in allowed else 'preflight_failed'
    if isinstance(error,ImportError):
        return 'native_import_failed'
    if isinstance(error,AssertionError):
        return 'kernel_numeric_mismatch'
    if isinstance(error,urllib.error.URLError):
        return 'native_fetch_failed'
    if isinstance(error,OSError):
        return 'native_environment_failed'
    return 'preflight_failed'

try:
    zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(root)
    sys.path.insert(0, str(root))
    import torch
    original_torch = importlib.metadata.version('torch')
    constraints = root / 'torch-constraint.txt'
    constraints.write_text('torch==' + original_torch + '\\n',encoding='utf-8')
    run_captured([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q',
        '-c',str(constraints),'-r',str(root / 'requirements-worker.txt')],600)
    if importlib.metadata.version('torch') != original_torch:
        raise RuntimeError('torch_changed')
    from contextlib import contextmanager, redirect_stdout, redirect_stderr
    import importlib
    from worker import Models, t4_efficient_sdpa, ensure_awq_import_compat, safe_job_diagnostic
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (7,5):
        raise RuntimeError('free_t4_required')
    stage = 'toolkit_check'
    from torch.utils.cpp_extension import CUDA_HOME, get_compiler_abi_compatibility_and_version
    if not CUDA_HOME:
        raise RuntimeError('native_toolkit_missing')
    nvcc = pathlib.Path(CUDA_HOME) / 'bin' / 'nvcc'
    if not nvcc.is_file() or not (pathlib.Path(CUDA_HOME) / 'include' / 'cuda_runtime.h').is_file():
        raise RuntimeError('native_toolkit_missing')
    if not (pathlib.Path(sysconfig.get_path('include')) / 'Python.h').is_file():
        raise RuntimeError('python_headers_missing')
    nvcc_output = run_captured([str(nvcc),'--version'],20)
    safe_toolkit_diagnostic(nvcc_output,torch.version.cuda)
    cuda_major,cuda_minor = toolkit_release(nvcc_output,torch.version.cuda)
    compiler = shutil.which('g++')
    if not compiler:
        raise RuntimeError('native_compiler_missing')
    with redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()):
        compatible,_ = get_compiler_abi_compatibility_and_version(compiler)
    if not compatible:
        raise RuntimeError('native_compiler_incompatible')
    print('recipe_preflight stage=toolkit_check result=ok cuda_major=' + str(cuda_major)
        + ' cuda_minor=' + str(cuda_minor) + ' python_major=' + str(sys.version_info.major)
        + ' python_minor=' + str(sys.version_info.minor),flush=True)
    stage = 'native_fetch'
    commit = '3d1413bf19a72451632f8dc2bb44c88bf2faed3e'
    source_url = 'https://codeload.github.com/casper-hansen/AutoAWQ_kernels/zip/' + commit
    with urllib.request.urlopen(source_url,timeout=30) as response:
        archive = response.read(20*1024*1024+1)
    if len(archive) > 20*1024*1024:
        raise RuntimeError('native_source_too_large')
    source_root = root / 'native-source'
    source_root.mkdir()
    with zipfile.ZipFile(io.BytesIO(archive)) as source_zip:
        for item in source_zip.infolist():
            if not (source_root / item.filename).resolve().is_relative_to(source_root.resolve()):
                raise RuntimeError('native_source_path_invalid')
        source_zip.extractall(source_root)
    source = source_root / ('AutoAWQ_kernels-' + commit)
    setup = source / 'setup.py'
    setup.write_text(select_awq_extension(setup.read_text(encoding='utf-8')),encoding='utf-8')
    stage = 'build_dependencies'
    run_captured([sys.executable,'-m','pip','install','--disable-pip-version-check','-q',
        '--no-deps','setuptools','wheel','numpy','ninja'],90)
    stage = 'native_build'
    build_env = dict(os.environ,MAX_JOBS='2',COMPUTE_CAPABILITIES='75',CC=compiler,CXX=compiler)
    # Do not let inherited metadata overrides misrepresent the actual toolkit/Torch.
    for key in ['CUDA_VERSION','TORCH_VERSION','PYPI_BUILD']:
        build_env.pop(key,None)
    wheels = root / 'native-wheels'
    wheels.mkdir()
    started = time.monotonic()
    run_captured([sys.executable,'-m','pip','wheel','--disable-pip-version-check','--no-deps',
        '--no-build-isolation','--wheel-dir',str(wheels),str(source)],420,env=build_env)
    built = list(wheels.glob('autoawq_kernels-*.whl'))
    if len(built) != 1 or importlib.metadata.version('torch') != original_torch:
        raise RuntimeError('native_build_invalid')
    print('recipe_preflight stage=native_build result=ok seconds=' + str(round(time.monotonic()-started,3)),flush=True)
    stage = 'native_install'
    run_captured([sys.executable,'-m','pip','install','--disable-pip-version-check','-q',
        '--no-deps',str(built[0])],60)
    if importlib.metadata.version('torch') != original_torch:
        raise RuntimeError('torch_changed')
    stage = 'kernel_check'
    ensure_awq_import_compat()
    native = importlib.import_module('awq_ext')
    gemm = importlib.import_module('awq.modules.linear.gemm')
    if not callable(getattr(native,'gemm_forward_cuda',None)) or not callable(getattr(native,'dequantize_weights_cuda',None)):
        raise RuntimeError('native_awq_unavailable')
    from awq.utils.packing_utils import dequantize_gemm
    @contextmanager
    def preflight_native_awq():
        original = (gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned)
        try:
            gemm.awq_ext, gemm.TRITON_AVAILABLE, gemm.user_has_been_warned = native, False, True
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
    with torch.inference_mode(), preflight_native_awq():
        restored = native.dequantize_weights_cuda(qweight,scales,qzeros,0,0,0,False)
        torch.testing.assert_close(restored,reference,rtol=0.001,atol=0.001)
        maximum_error = 0.0
        for rows in [1,16]:
            matrix = torch.randn((rows,512),device='cuda:0',dtype=torch.float16)
            result = native.gemm_forward_cuda(matrix,qweight,scales,qzeros,8)
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
        with torch.inference_mode(), t4_efficient_sdpa(), preflight_native_awq():
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
    print('recipe_preflight stage=' + stage + ' result=failed reason=' + safe_failure_reason(error),flush=True)
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
