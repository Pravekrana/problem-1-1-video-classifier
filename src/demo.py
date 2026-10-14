"""Generate a deterministic smoke-test video; not a real accuracy benchmark."""
import cv2, numpy as np
from pathlib import Path
p=Path("demo.avi")
writer=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"MJPG"),8,(160,128))
if not writer.isOpened(): raise RuntimeError("MJPG codec unavailable")
for i in range(24):
    frame=np.zeros((128,160,3),dtype=np.uint8)
    cv2.circle(frame,(20+4*i,64),16,(255,255,255),-1)
    writer.write(frame)
writer.release()
print(p)
