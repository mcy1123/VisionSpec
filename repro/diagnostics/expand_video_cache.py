#!/usr/bin/env python3
"""Duplicate + upscale frames in the frozen 16-frame video cache for 5-10k token tests."""
import argparse, cv2, pandas as pd
from pathlib import Path

def convert(row, out_dir, factor, scale):
    src=Path(row['cache_path']); dst=Path(out_dir)/src.name
    if dst.exists(): return str(dst)
    cap=cv2.VideoCapture(str(src)); fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames=[]
    while True:
        ok, frame=cap.read()
        if not ok: break
        if scale != 1.0:
            frame=cv2.resize(frame,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        frames.extend([frame]*factor)
    cap.release(); h,w=frames[0].shape[:2]
    writer=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    for f in frames: writer.write(f)
    writer.release(); return str(dst)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--pool',required=True); ap.add_argument('--out-dir',required=True); ap.add_argument('--out-parquet',required=True); ap.add_argument('--factor',type=int,default=2); ap.add_argument('--scale',type=float,default=2.0)
    a=ap.parse_args(); d=pd.read_parquet(a.pool); Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    d['cache_path']=[convert(r,a.out_dir,a.factor,a.scale) for r in d.to_dict('records')]
    d.to_parquet(a.out_parquet,index=False); print('wrote',a.out_parquet,len(d))
if __name__=='__main__': main()
