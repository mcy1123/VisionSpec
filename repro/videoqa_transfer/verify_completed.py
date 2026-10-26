"""Independent CPU verification of stored full outputs and pilot-to-full reproducibility."""
import pathlib,json,collections,statistics,sys,hashlib,zlib
R=pathlib.Path(sys.argv[1]);output={}
for sub in ['', 'msrvtt','nextqa_mc']:
 d=R/sub;gate=d/'full/gates.json'
 if not gate.exists() or not json.loads(gate.read_text()).get('pass'):continue
 raw=list(map(json.loads,(d/'full/results.jsonl').read_text().splitlines()));ar={r['input_key']:r for r in map(json.loads,(d/'full/ar.jsonl').read_text().splitlines())};pilot={r['input_key']:r for r in map(json.loads,(d/'smoke/ar.jsonl').read_text().splitlines())};groups=collections.defaultdict(list);rounds=0
 for r in raw:
  groups[r['input_key']].append(r);z=r['diagnostic'];ref=ar[r['input_key']]['tokens'];assert z['tokens']==ref
  assert all(x['tokens']==ref for x in r['clean_repeats'])
  assert abs(statistics.mean(x['accepted']+1 for x in z['rounds'])-z['stats']['tau'])<1e-12
  assert z['stats']['rounds']==len(z['rounds'])==len(z['trees'])
  assert sum(x['new_root']+x['accepted_effective']+x['bonus_emitted'] for x in z['clipped_rounds'])==len(ref)
  assert all(len(t['nodes'])<=63 for t in z['trees']);rounds+=len(z['rounds'])
 assert all(len(v)==3 and {r['arm'] for r in v}=={'C0','C1','C2'} and len({r['diagnostic']['first_context_hash'] for r in v})==1 for v in groups.values())
 assert len(groups)==len(ar)
 assert all(k in ar and x['tokens']==ar[k]['tokens'] for k,x in pilot.items())
 resume_audit=None
 if (d/'full/resume_provenance.json').exists():
  prev=pathlib.Path(json.loads((d/'full/resume_provenance.json').read_text())['source'])
  old=list(map(json.loads,(prev/'results.jsonl').read_text().splitlines()))
  assert [{k:v for k,v in r.items() if k!='execution_segment'} for r in raw[:len(old)]]==old
  pi={r['key']:r['hashes'] for r in map(json.loads,(prev/'inputs.jsonl').read_text().splitlines())}
  ni={r['key']:r['hashes'] for r in map(json.loads,(d/'full/inputs.jsonl').read_text().splitlines())}
  assert pi==ni
  assert {r['input_key']:r for r in map(json.loads,(prev/'ar.jsonl').read_text().splitlines())}==ar
  assert all(len({r['execution_segment'] for r in group})==1 for group in groups.values())
  resume_audit={'pass':True,'reused_records_unchanged':len(old),'fresh_input_hashes_identical':len(ni),'all_AR_records_unchanged':len(ar),'all_arms_within_same_segment':True}
 for k,x in ar.items():assert all(c['tokens']==x['tokens'] for c in x['clean_repeats'])
 first_key=next(iter(ar));assert len(ar[first_key]['tokens'])<=16
 warmup_head_rounds=sum(r['diagnostic']['stats']['rounds'] for r in groups[first_key])
 calls=json.loads(gate.read_text())['calls']
 warm_cycles=calls['SPEC_warmup_sequences']//3
 assert calls['SPEC_warmup_sequences']==3*warm_cycles
 expected_head=3*rounds+warm_cycles*warmup_head_rounds
 expected_target=3*sum(len(x['tokens']) for x in ar.values())+calls['AR_warmup_sequences']*len(ar[first_key]['tokens'])+expected_head+3*len(raw)+calls['SPEC_warmup_sequences']
 assert calls['head_forwards']==expected_head and calls['target_forwards']==expected_target
 manifest=list(map(json.loads,(d/'dataset_manifest_20videos.jsonl').read_text().splitlines()));videos={x['video_id']:x for x in manifest}
 assert len({(x['video_id'],x['question_id'],x['prompt_mode']) for x in manifest})==len(manifest)
 for vid,x in videos.items():
  raw_bytes=pathlib.Path(x['raw_path']).read_bytes();assert hashlib.sha256(raw_bytes).hexdigest()==x['hash']
  assert hashlib.sha256(pathlib.Path(x['cache_path']).read_bytes()).hexdigest()==x['cache_hash']
  assert len(x['sample_frame_ids'])==len(x['sample_timestamps'])==16
  assert x['sample_timestamps']==sorted(x['sample_timestamps'])
  assert max(x['sample_frame_ids'])<x['source_frame_count']
  assert len({y['question_id'] for y in manifest if y['video_id']==vid})<=2
  if 'zip_member' in x:
   assert len(raw_bytes)==x['zip_member']['size'] and zlib.crc32(raw_bytes)==x['zip_member']['CRC']
  else:assert len(raw_bytes)==x['download_range']['size']
 output[sub or 'MSVD']={'pass':True,'raw_and_cache_hash_verified_videos':len(videos),'full_inputs':len(ar),'full_arm_records':len(raw),'independent_SPEC_AR_checks':len(raw)*3,'rounds_recomputed':rounds,'verified_target_forwards':expected_target,'verified_head_forwards':expected_head,'pilot_full_AR_identical':len(pilot),'cross_arm_context_identical':len(groups),'resume_audit':resume_audit,'note':'CPU audit of existing raw evidence; no new model calls'}
(R/'independent_evidence_verification.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
