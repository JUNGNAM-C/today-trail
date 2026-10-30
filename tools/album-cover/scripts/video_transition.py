import os
os.environ['STEP']='0.8'; os.environ['DURK']='1.1'
_ns={}
exec(open('jn/jna.py').read().split("if os.environ.get('TS')")[0],_ns)
_ns['rm_f'][:]=0; _ns['T0']=0.3
import numpy as np, cv2, os
from PIL import Image, ImageDraw, ImageFont
A='/mnt/user-data/attachments/5a7d1354-b855-5f9c-b6ac-3454f8809492'
P=np.load('jn/T_prev.npz'); C=np.load('jn/T_cur.npz')
S=840; CW,CH=1080,1920; sx=(CW-S)//2; sy=480
def rs(a,interp=cv2.INTER_AREA): return cv2.resize(a,(S,S),interpolation=interp)
pp=rs(P['paint']).astype(np.float32); cp=rs(C['paint']).astype(np.float32)
pm=rs(P['runner'].astype(np.float32)*255,cv2.INTER_LINEAR)/255.; cm=rs(C['runner'].astype(np.float32)*255,cv2.INTER_LINEAR)/255.
pm=np.clip(pm,0,1); cm=np.clip(cm,0,1)
PCOL=np.array([26,30,56],np.float32); CCOL=np.array([42,26,48],np.float32)
def bbox(m):
    ys,xs=np.where(m>0.5); return xs.min(),xs.max(),ys.min(),ys.max()
px0,px1,py0,py1=bbox(pm); cx0,cx1,cy0,cy1=bbox(cm)
pc=np.array([(px0+px1)/2,(py0+py1)/2]); cc=np.array([(cx0+cx1)/2,(cy0+cy1)/2])
sfin=max((cx1-cx0)/(px1-px0),(cy1-cy0)/(py1-py0)); print('scale',sfin)
# runner grain so sprites match painted texture
rng=np.random.RandomState(4); gr=cv2.GaussianBlur(rng.randn(S,S).astype(np.float32),(0,0),0.8)*4
ps=P['split']-190+sy; cs=C['split']-190+sy
ptop,pbot=P['topc'].astype(np.float32),P['botc'].astype(np.float32)
ctop,cbot=C['topc'].astype(np.float32),C['botc'].astype(np.float32)
# static overlay (title/label)
ov=Image.new('RGBA',(CW,CH),(0,0,0,0)); d=ImageDraw.Draw(ov)
def spaced(text,y,size,fill,fp,fit):
    f=ImageFont.truetype(fp,size); ws=[d.textlength(c,font=f) for c in text]
    gap=(fit-sum(ws))/(len(text)-1); x=(CW-fit)/2
    for c,w in zip(text,ws): d.text((x,y),c,font=f,fill=fill); x+=w+gap
spaced('KILIAN JORNET',sy-100,44,(245,238,222,255),A+'/3ab4a850-WantedSans-Bold.ttf',840)
spaced('RICH ROLL PODCAST',1690,26,(36,52,78,255),A+'/5a27bef6-WantedSans-Medium.ttf',470)
ov=np.array(ov).astype(np.float32); ova=ov[...,3:4]/255.
msk=Image.new('L',(CW,CH),0); ImageDraw.Draw(msk).rounded_rectangle([0,0,CW-1,CH-1],40,fill=255)
msk=np.array(msk).astype(np.float32)[...,None]/255.
corner=np.array([12,22,48],np.float32)
def sm(t,a,b): 
    x=np.clip((t-a)/(b-a),0,1); return x*x*(3-2*x)
def eio4(x): x=np.clip(x,0,1); return 8*x**4 if x<.5 else 1-8*(1-x)**4
yy=np.arange(CH)[:,None]
def eo3(x): x=np.clip(x,0,1); return 1-(1-x)**3
def thick(m):
    dt=cv2.distanceTransform((m>0.5).astype(np.uint8),cv2.DIST_L2,5); y,x=np.unravel_index(dt.argmax(),dt.shape); return np.array([x,y],np.float32)
def cover_scale(m,c):
    lo,hi=2.0,200.0
    for _ in range(18):
        sc=(lo+hi)/2
        M=np.array([[sc,0,S/2-sc*c[0]],[0,sc,S/2-sc*c[1]]],np.float32)
        w=cv2.warpAffine(m,M,(S,S),flags=cv2.INTER_LINEAR,borderValue=0)
        if w.min()>0.97: hi=sc
        else: lo=sc
    return hi*1.05
pm_anchor0=None
tp=thick(pm); tc=thick(cm); sp=cover_scale(pm,tp); scc=cover_scale(cm,tc)
print('thick',tp,tc,'cover',sp,scc)
mid=np.array([S/2,S/2],np.float32); pm_anchor0=tp.copy()
T0Z=6.0; TB=6.75; TE=7.5
def warp(m,sc,anchor_src,anchor_dst):
    M=np.array([[sc,0,anchor_dst[0]-sc*anchor_src[0]],[0,sc,anchor_dst[1]-sc*anchor_src[1]]],np.float32)
    return cv2.warpAffine(m,M,(S,S),flags=cv2.INTER_LINEAR,borderValue=0)
def frame(t):
    k=sm(t,T0Z,7.0)
    if t<7.0:
        c=_ns['frame2'](t); pp_t=c[sy:sy+S,sx:sx+S].astype(np.float32)
    else: pp_t=pp
    top=ptop*(1-k)+ctop*k; bot=pbot*(1-k)+cbot*k; split=ps*(1-k)+cs*k
    w=np.clip(yy-split+0.5,0,1)
    bg=top[None,None,:]*(1-w[...,None])+bot[None,None,:]*w[...,None]
    bg=np.broadcast_to(bg,(CH,CW,3)).copy()
    sq=pp_t*(1-k)+cp*k
    if t<TB:
        u=np.clip((t-T0Z)/(TB-T0Z),0,1); e=u**2.6
        sc=np.exp(np.log(sp)*e)
        anc=pc*(1-e)+mid*e if False else pm_anchor0*(1-e)+mid*e
        bob=np.sin(2*np.pi*t*0.35)*2.0*0.84*(1-e)
        m1=warp(pm,sc,tp,anc+np.array([0,bob],np.float32))
        col=PCOL*(1-e)+CCOL*e
        sq=sq*(1-m1[...,None])+(col[None,None,:]+gr[...,None])*m1[...,None]
    else:
        v=eo3((t-TB)/(TE-TB))
        sc=np.exp(np.log(scc)*(1-v))
        anc=mid*(1-v)+tc*v
        m2=warp(cm,sc,tc,anc)
        sq=sq*(1-m2[...,None])+(CCOL[None,None,:]+gr[...,None])*m2[...,None]
    bg[sy:sy+S,sx:sx+S]=np.clip(sq,0,255)
    out=bg*(1-ova)+ov[...,:3]*ova
    out=out*msk+corner*(1-msk)
    return np.clip(out,0,255).astype(np.uint8)
if os.environ.get('TS'):
    ts=[float(v) for v in os.environ['TS'].split(',')]
    ims=[cv2.resize(frame(t),(360,640)) for t in ts]
    Image.fromarray(np.hstack(ims)).save('jn/Tsheet3.png')
else:
    os.makedirs('jn/framesT3',exist_ok=True)
    for i in range(405):
        Image.fromarray(frame(i/30)).save('jn/framesT3/f_%04d.jpg'%i,quality=95)
