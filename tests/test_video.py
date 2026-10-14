import cv2,numpy as np,pytest
from src.main import sample
def test_sampling(tmp_path):
    p=tmp_path/"test.avi"
    w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*"MJPG"),8,(64,48))
    assert w.isOpened()
    for i in range(4): w.write(np.full((48,64,3),i*50,dtype=np.uint8))
    w.release()
    clip,count=sample(p)
    assert clip.shape==(3,16,128,128)
    assert count==4
    assert clip.min()>=0 and clip.max()<=1
def test_missing(tmp_path):
    with pytest.raises(ValueError): sample(tmp_path/"missing.mp4")
