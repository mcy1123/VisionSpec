"""Actual acquired-video inventory; separate video bytes from HTTP metadata/retry traffic."""
import pathlib,json,csv,statistics,collections,sys
R=pathlib.Path(sys.argv[1]);out=[];vv=[]
for sub in ['', 'msrvtt','nextqa_mc']:
 d=R/sub;p=d/'dataset_manifest_20videos.jsonl'
 if not p.exists():p=d/'dataset_manifest_3videos.jsonl'
 if not p.exists():continue
 records=list(map(json.loads,p.read_text().splitlines()));videos={x['video_id']:x for x in records};qs={(x['video_id'],x['question_id']):x for x in records};dur=[x['video_duration'] for x in videos.values()];audio=sum(any(s['type']=='audio' for s in x['source_probe']['streams']) for x in videos.values());m=json.loads((d/'sampling_manifest.json').read_text());row={'dataset':records[0]['dataset'],'setting':'MC' if records[0].get('options') else 'open-ended','split':'val','frozen_videos':len(m['video_ids']),'acquired_videos':len(videos),'acquired_QA':len(qs),'raw_video_bytes':sum(pathlib.Path(x['raw_path']).stat().st_size for x in videos.values()),'cache_bytes':sum(pathlib.Path(x['cache_path']).stat().st_size for x in videos.values()),'duration_mean_s':statistics.mean(dur),'duration_min_s':min(dur),'duration_max_s':max(dur),'fps_values':sorted(set(x['source_fps'] for x in videos.values())),'audio_stream_videos':audio,'audio_used':False,'question_type_counts':dict(collections.Counter(x['question_type'] for x in qs.values())),'all_16_frames':all(len(x['sample_frame_ids'])==16 for x in videos.values()),'minimum_unique_sample_frames':min(len(set(x['sample_frame_ids'])) for x in videos.values()),'frozen_at':m['frozen_at'],'byte_note':'Unique acquired video bytes, not total HTTP traffic including range overread/metadata/retries'};out.append(row)
 for vid,x in videos.items():vv.append({'dataset':x['dataset'],'video_id':vid,'source_id':x['source_id'],'overlap_status':x['overlap_status'],'raw_video_bytes':pathlib.Path(x['raw_path']).stat().st_size,'video_duration':x['video_duration'],'source_fps':x['source_fps'],'source_frame_count':x['source_frame_count'],'hash':x['hash'],'cache_hash':x['cache_hash']})
(R/'dataset_inventory.json').write_text(json.dumps(out,indent=2));
with (R/'video_inventory.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(vv[0]));w.writeheader();w.writerows(vv)
print(json.dumps(out,indent=2))
