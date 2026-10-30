import os, sys, numpy as np, cv2, scipy.ndimage as ndi, math
from PIL import Image, ImageDraw, ImageFont
exec(open('jn/jn2.py').read())          # static painting -> paint, G, runner, ridge, bands, ...
SAMPLE=os.environ.get('SAMPLE')
FPS=30; DUR=float(os.environ.get('DUR','8')); NF=int(FPS*DUR)
def pn(sx,sy,seed):
    r=np.random.RandomState(seed).randn(N,N).astype(np.float32)
    g=ndi.gaussian_filter(r,(sy,sx),mode='wrap'); return (g/g.std()).astype(np.float32)
# periodic fields (loop seamlessly)
swell=pn(120,46,1)                      # long swell: broad undulation
ripple=pn(55,22,2)                      # finer ripple
s1=pn(140,1.2,3); s2=pn(70,2.4,4); s3=pn(220,3.5,5)
straight=np.zeros((N,N,3),np.float32)
for y in range(N):
    k=0
    for i,b in enumerate(ys):
        if y>=b: k=i
    straight[y]=cols[k]
Y=np.arange(N,dtype=np.float32)[:,None]; X=np.tile(np.arange(N,dtype=np.float32),(N,1))
gn=ndi.gaussian_filter(np.random.RandomState(9).randn(N,N).astype(np.float32),0.7)
def shift(a,dx): 
    dx=int(round(dx))%N
    return np.roll(a,dx,axis=1)
def sky_frame(t):
    ph=t/DUR                              # 0..1 loop
    # swells travel to the right and breathe up/down
    d=shift(swell,N*ph)*7.5+shift(ripple,-N*2*ph)*3.0
    d=d*(0.85+0.3*math.sin(2*math.pi*ph*2))
    mp=(Y+d).astype(np.float32)
    sk=cv2.remap(straight,X,mp,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    sk=cv2.GaussianBlur(sk,(0,0),1.2)
    a=cv2.remap(shift(s1,N*ph),X,mp,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    b=cv2.remap(shift(s2,-N*2*ph),X,mp,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    c=cv2.remap(shift(s3,N*ph),X,mp,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    sk=sk*(1+(0.016*a+0.010*b+0.012*c)*amp)[...,None]
    sk+= (gn*2.2)[...,None]
    return np.clip(sk,0,255)
# runner geometry
rm=runner.astype(np.uint8)
ysr,xsr=np.where(rm>0)
foot_y=ysr.max(); foot_x=int(xsr[ysr>foot_y-14].mean())
rs_=ndi.gaussian_filter1d(ridge.astype(np.float32),30)
cx0=int(xsr.mean()); cy0=int(ysr.mean())
RUN_COL=np.array([26,30,56],np.float32)
X0=float(os.environ.get('X0','-250')); X1=float(os.environ.get('X1','330'))
Hm=float(ysr.max()-ysr.min()); Wm=float(xsr.max()-xsr.min())
print('runner H,W',Hm,Wm)
SS=2
def capsule(m,p,q,r1,r2):
    p=np.array(p)*SS; q=np.array(q)*SS; r1*=SS; r2*=SS
    d=q-p; L=np.hypot(*d)+1e-6; n=np.array([-d[1],d[0]])/L
    poly=np.array([p+n*r1,q+n*r2,q-n*r2,p-n*r1],np.float32).astype(np.int32)
    cv2.fillConvexPoly(m,poly,255,cv2.LINE_AA)
    cv2.circle(m,tuple(int(v) for v in p),int(round(r1)),255,-1,cv2.LINE_AA)
    cv2.circle(m,tuple(int(v) for v in q),int(round(r2)),255,-1,cv2.LINE_AA)
def vec(a_deg,L): 
    a=math.radians(a_deg); return np.array([math.sin(a)*L,math.cos(a)*L])   # angle from straight-down, forward(+x)=positive
from scipy.interpolate import CubicSpline
KF_T=np.array([40,15,-15,-35,-20,15,55,52,40],float)      # thigh angle (deg from straight down, forward +) over one stride
KF_K=np.array([30,20,15,35,105,110,90,50,30],float)       # knee flexion
KX=np.arange(9,dtype=float)
CS_T=CubicSpline(KX,KF_T,bc_type='periodic'); CS_K=CubicSpline(KX,KF_K,bc_type='periodic')
def gait(ph):
    u=((ph/(2*math.pi))%1.0)*8.0
    return float(CS_T(u)), float(np.clip(CS_K(u),5,125))
def leg(m,hip,ph,H):
    th=H*0.245; sh=H*0.245
    ang,kn=gait(ph)
    knee=hip+vec(ang,th)
    ank=knee+vec(ang-kn,sh)
    toe=ank+vec(ang-kn+82,H*0.105)
    capsule(m,hip,knee,H*0.066,H*0.045)
    capsule(m,knee,ank,H*0.045,H*0.030)
    capsule(m,ank,toe,H*0.036,H*0.028)
def arm(m,sho,ph,H):
    ua=H*0.17; fa=H*0.155
    ang0,_=gait(ph+math.pi)                 # opposite leg drives the arm
    ang=-0.95*ang0+4
    elb=sho+vec(ang,ua)
    fl=95-8*math.sin(ph)+ (10 if ang>0 else 0)
    wr=elb+vec(ang+fl,fa)
    capsule(m,sho,elb,H*0.040,H*0.032)
    capsule(m,elb,wr,H*0.032,H*0.026)
    cv2.circle(m,tuple(int(v) for v in wr*SS),int(H*0.028*SS),255,-1,cv2.LINE_AA)
def pose_mask(cx,hip_y,t,H):
    m=np.zeros((N*SS,N*SS),np.uint8)
    ph=2*math.pi*1.15*t
    bounce=-H*0.045*(1-math.cos(2*ph+1.0))/2
    hip=np.array([cx,hip_y+bounce],np.float64)
    lean=20
    tor=H*0.31
    sho=hip+np.array([math.sin(math.radians(lean))*tor,-math.cos(math.radians(lean))*tor])
    # far limbs first (same colour; order only matters for overlap)
    leg(m,hip,ph+math.pi,H); arm(m,sho,ph,H)
    capsule(m,hip,sho,H*0.062,H*0.082)
    capsule(m,sho+np.array([H*0.006,-H*0.01]),sho+np.array([H*0.02,-H*0.05]),H*0.024,H*0.022)  # neck
    head=sho+np.array([H*0.045,-H*0.085])
    cv2.circle(m,tuple(int(v) for v in head*SS),int(H*0.062*SS),255,-1,cv2.LINE_AA)
    capsule(m,head+np.array([-H*0.03,-H*0.035]),head+np.array([-H*0.085,-H*0.01]),H*0.03,H*0.018)  # hair
    leg(m,hip,ph,H); arm(m,sho,ph+math.pi,H)
    m=cv2.resize(m,(N,N),interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
    return m
hip_ref=float(ysr.max())-Hm*0.50
cx_ref=float(xsr.mean())-Hm*0.04
def runner_frame(img,t):
    ph=t/DUR
    e=ph*ph*(3-2*ph)*0.35+ph*0.65
    dx=X0+(X1-X0)*e
    fx=int(np.clip(foot_x+dx,0,N-1))
    dy=(rs_[fx]-rs_[foot_x])
    m=pose_mask(cx_ref+dx,hip_ref+dy,t,Hm*1.02)
    m=cv2.GaussianBlur(m,(0,0),0.5)
    return img*(1-m[...,None])+RUN_COL*m[...,None]
# canvas
CW,CH=1080,1920; S=840; sx=(CW-S)//2; sy=480
topc_=topc; botc_=botc
split=int(round(sy+np.median(ridge)*S/N))
base=Image.new('RGB',(CW,CH),topc_); dd=ImageDraw.Draw(base)
dd.rectangle([0,split,CW,CH],fill=botc_)
def spaced2(text,y,size,fill,fontpath,fit):
    f=ImageFont.truetype(fontpath,size); ws=[dd.textlength(c,font=f) for c in text]
    gap=(fit-sum(ws))/(len(text)-1); x=(CW-fit)/2
    for c,w in zip(text,ws): dd.text((x,y),c,font=f,fill=fill); x+=w+gap
A='/mnt/user-data/attachments/5a7d1354-b855-5f9c-b6ac-3454f8809492'
spaced2('KILIAN JORNET',sy-100,44,(245,238,222),A+'/3ab4a850-WantedSans-Bold.ttf',840)
spaced2('RICH ROLL PODCAST',1690,26,(36,52,78),A+'/5a27bef6-WantedSans-Medium.ttf',470)
msk=Image.new('L',(CW,CH),0); ImageDraw.Draw(msk).rounded_rectangle([0,0,CW-1,CH-1],40,fill=255)
bgc=Image.new('RGB',(CW,CH),(12,22,48)); bgc.paste(base,(0,0),msk); base=bgc
base_np=np.array(base)
os.makedirs('jn/frames',exist_ok=True)
def frame(t):
    sq=paint.astype(np.float32).copy()
    sk=sky_frame(t); sq[~G]=sk[~G]
    sq=runner_frame(sq,t)
    sq=cv2.resize(np.clip(sq,0,255).astype(np.uint8),(S,S),interpolation=cv2.INTER_AREA)
    out=base_np.copy(); out[sy:sy+S,sx:sx+S]=sq
    return out
if SAMPLE:
    ts=[float(v) for v in SAMPLE.split(',')]
    tiles=[Image.fromarray(frame(t)).resize((360,640)) for t in ts]
    sh=Image.new('RGB',(360*len(tiles),640))
    for i,tl in enumerate(tiles): sh.paste(tl,(360*i,0))
    sh.save('jn/vsample.png')
else:
    for i in range(NF):
        Image.fromarray(frame(i/FPS)).save(f'jn/frames/f_{i:04d}.jpg',quality=92)
        if i%40==0: print(i,flush=True)
