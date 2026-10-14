"""Reproducible video action recognition baseline (Kinetics-400 R3D-18)."""
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np

def sample(path, n=16, size=128):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise ValueError(f"Cannot open {path}")
    frames=[]
    try:
        while True:
            ok, frame=cap.read()
            if not ok: break
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w=frame.shape[:2]
            scale=size/min(h,w)
            frame=cv2.resize(frame,(max(size,round(w*scale)),max(size,round(h*scale))))
            y=(frame.shape[0]-size)//2; x=(frame.shape[1]-size)//2
            frames.append(frame[y:y+size,x:x+size])
    finally: cap.release()
    if not frames: raise ValueError(f"No frames: {path}")
    idx=np.linspace(0,len(frames)-1,n).astype(int)
    clip=np.stack([frames[i] for i in idx]).astype("float32")/255
    return np.transpose(clip,(3,0,1,2)),len(frames)

def predict(path,model,labels,device):
    import torch
    start=time.perf_counter()
    clip,count=sample(path)
    mean=np.array([.43216,.394666,.37645],dtype="float32")[:,None,None,None]
    std=np.array([.22803,.22145,.216989],dtype="float32")[:,None,None,None]
    x=torch.from_numpy(((clip-mean)/std).copy()).unsqueeze(0).to(device)
    with torch.inference_mode():
        scores=torch.softmax(model(x),dim=1)[0]
        values,indices=torch.topk(scores,5)
    return {"video":str(path),"frames":count,"latency_seconds":round(time.perf_counter()-start,4),"predictions":[{"label":labels[int(i)],"score":round(float(v),5)} for v,i in zip(values.cpu(),indices.cpu())]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--output",default="results.json")
    p.add_argument("--labels",help="CSV columns video,label with Kinetics-400 labels")
    p.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    a=p.parse_args()
    import torch
    from torchvision.models.video import r3d_18,R3D_18_Weights
    if a.device=="cuda" and not torch.cuda.is_available(): p.error("CUDA unavailable")
    weights=R3D_18_Weights.DEFAULT
    model=r3d_18(weights=weights).to(a.device).eval()
    path=Path(a.input)
    videos=[path] if path.is_file() else sorted(f for f in path.rglob("*") if f.suffix.lower() in {".mp4",".avi",".mov",".mkv",".webm"})
    if not videos: p.error("No video files found")
    results=[]
    for video in videos:
        try: result=predict(video,model,weights.meta["categories"],a.device)
        except (ValueError,cv2.error) as e: result={"video":str(video),"error":str(e)}
        print(json.dumps(result),flush=True); results.append(result)
    report={"model":"R3D-18 Kinetics-400","results":results}
    if a.labels:
        import csv
        with open(a.labels,newline="",encoding="utf8") as f:
            labels={row["video"]:row["label"] for row in csv.DictReader(f)}
        matched=[(labels.get(r["video"],labels.get(Path(r["video"]).name)),r["predictions"]) for r in results if "predictions" in r]
        matched=[(truth,preds) for truth,preds in matched if truth is not None]
        report["evaluation"]={"count":len(matched),"top1":sum(truth==preds[0]["label"] for truth,preds in matched)/len(matched) if matched else None,"top5":sum(truth in [p["label"] for p in preds] for truth,preds in matched)/len(matched) if matched else None}
    Path(a.output).write_text(json.dumps(report,indent=2))
if __name__=="__main__": main()
