"""Verify pinned weights against official Hub metadata and isolate the files."""
import hashlib,json,os,shutil,time,traceback
from pathlib import Path
os.environ.pop('HF_HUB_OFFLINE',None)
os.environ.pop('TRANSFORMERS_OFFLINE',None)
os.environ['HF_HUB_DISABLE_XET']='1'
from huggingface_hub import hf_hub_download
from transformers.utils.hub import cached_file
from prepare import ROOT,atomic,sha

def main():
    meta=json.loads((ROOT/'official_model_metadata.json').read_text());revision=meta['sha']
    assert meta['id']=='Qwen/Qwen3-0.6B' and revision=='c1899de289a04d12100db370d81485cdf75e47ca'
    entries={r['rfilename']:r for r in meta['siblings']}
    expected=entries['model.safetensors']['lfs']['sha256']
    target=ROOT/'verified_base';target.mkdir(exist_ok=True)
    files={}
    for name in ('model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json'):
        entry=entries[name]
        source=Path(cached_file('Qwen/Qwen3-0.6B',name,revision=revision,local_files_only=True));digest=sha(source)
        if 'lfs' in entry:assert digest==entry['lfs']['sha256'],name
        else:
            data=source.read_bytes();gitsha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert gitsha==entry['blobId'],(name,gitsha,entry['blobId'])
        assert source.stat().st_size==entry['size']
        dest=target/name
        if not dest.exists() or sha(dest)!=digest:shutil.copy2(source,dest)
        assert sha(dest)==digest
        files[name]={'sha256':digest,'path':str(dest),'official_blob':entry,'source':str(source)}
    atomic(ROOT/'BASE_PROVENANCE.json',{'repository':'Qwen/Qwen3-0.6B','revision':revision,'model_directory':str(target),
        'expected_model_sha256':expected,'verified':True,'initial_adapters':[],'files':files,
        'verification':'Compare official LFS SHA256 / Git blob SHA1 and size, not the local cache filename.',
        'metadata_url':f'https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/{revision}?blobs=true'})
    atomic(ROOT/'BASE_AUDIT_CORRECTION.json',{'verified':True,'shared_base_matches_official':True,
        'explanation':'Initial audit incorrectly assumed cache filename equals LFS SHA256. Official pinned API metadata confirms f47f... SHA256. Original shared weights are valid and untouched.',
        'restart_policy':'Archive short pre-audit training and restart from verified isolated identical weights.'})
    print('BASE_VERIFIED',expected,flush=True)
if __name__=='__main__':
    try:main()
    except BaseException:
        atomic(ROOT/'BASE_DOWNLOAD_FAILED.json',{'traceback':traceback.format_exc()});raise
