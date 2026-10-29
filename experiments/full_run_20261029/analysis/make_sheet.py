import sys, imageio.v2 as iio, numpy as np
from PIL import Image
out=sys.argv[1]
rows=[]
for f in sys.argv[2:]:
    fr=iio.mimread(f, memtest=False); n=len(fr)
    idx=np.linspace(0,n-1,8).astype(int)
    row=np.concatenate([np.array(Image.fromarray(fr[i]).resize((160,160))) for i in idx],1)
    rows.append(row); print(f, n)
Image.fromarray(np.concatenate(rows,0)).save(out, quality=85)
