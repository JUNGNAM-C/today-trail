import os
os.environ['SAMPLE']=''
os.environ['DUR']=os.environ.get('DUR','15')
exec(open('jn/jnv.py').read().split("if SAMPLE:")[0])
FILL=float(os.environ.get('FILL','6.5'))
PAPER=np.array([245,238,222],np.float32)
Xc=np.arange(N,dtype=np.float32)
col_noise=ndi.gaussian_filter1d(np.random.RandomState(21).randn(N*2).astype(np.float32),6,mode='wrap')
col_noise=col_noise/col_noise.std()
rm_f=cv2.GaussianBlur(runner.astype(np.float32),(0,0),0.8)
def smooth(a): return a*a*(3-2*a)
def front(t,lag):
    e=np.clip((t-lag)/FILL,0,1); e=smooth(e)
    top=-40.0                                   # slightly beyond the top edge
    base=rs_.astype(np.float32)                 # starts at the ridge
    f=base+(top-base)*e
    A=34*(1-0.55*e)*np.sin(np.pi*min(1,max(0,(t-lag)/FILL)))**0.5 if t>lag else 0
    w=0.6*np.sin(2*np.pi*Xc/460-2*np.pi*0.55*t)+0.4*np.sin(2*np.pi*Xc/210+2*np.pi*0.9*t+1.0)
    jit=7*np.roll(col_noise[:N],int(t*40))
    return f+ A*w + jit
NB=len(ys)                                  # 6 colour layers
ye=np.array(list(ys)+[700.0],np.float32)
STEP=float(os.environ.get('STEP','2.0')); DURK=float(os.environ.get('DURK','2.5')); T0=0.5
RAG=np.random.RandomState(5).uniform(-60,60,(NB,24)).astype(np.float32)
RAG=ndi.gaussian_filter1d(RAG,1.0,axis=1)
def disp_t(t):
    ph=t/DUR
    d=shift(swell,N*ph)*7.5+shift(ripple,-N*2*ph)*3.0
    return d*(0.85+0.3*math.sin(2*math.pi*ph*2))
def prog(t,s):
    p=np.clip((t-T0-s*STEP)/DURK,0,1); return p*p*(3-2*p)*0.30+p*0.70
def frame2(t):
    sq=paint.astype(np.float32).copy()
    sk=sky_frame(t)
    mp=(Y+disp_t(t)).astype(np.float32)             # source row of every pixel (same undulation as the sky)
    mpb=np.broadcast_to(mp,(N,N))
    mpb=mpb+0*X
    idx=np.clip(np.searchsorted(np.array(ys,np.float32),mpb,side='right')-1,0,NB-1)
    hband=(ye[idx+1]-ye[idx])
    frac=np.clip((mpb-ye[idx])/hband,0,0.9999)
    s_order=(NB-1-idx)                              # bottom layer is painted first
    p=prog(t,s_order.astype(np.float32))
    dirn=np.where(s_order%2==0,1.0,-1.0)            # left->right, right->left, ...
    xp=np.where(dirn>0,X,N-X)
    head=p*(N+90)
    jj=np.clip((frac*24).astype(np.int32),0,23)
    rag=RAG[idx,jj]*(1-p)**0.4*(p>0)
    soft=9.0+16.0*np.abs(RAG[idx,jj])/60.0
    cover=np.clip((head-rag-xp)/soft,0,1)
    done_t=T0+s_order*STEP+DURK
    sett=np.clip((t-done_t)/1.8,0,1); sett=sett*sett*(3-2*sett)
    lead=np.exp(-np.abs(head-xp)/8.0)*(p>0)*(p<1)
    streak=0.035*np.sin(frac*55+idx*1.7)+0.02*np.sin(frac*21+idx*3.1)
    tone=1+streak*(1-sett)
    # drier, slightly lighter bristle marks while the stroke is fresh
    col=sk*tone[...,None]*(1-0.09*lead)[...,None]
    base_p=PAPER+(gn*2.5)[...,None]
    out=base_p*(1-cover[...,None])+col*cover[...,None]
    sq[~G]=np.clip(out,0,255)[~G]
    bob=math.sin(2*math.pi*t*0.35)*2.0
    M=np.float32([[1,0,0],[0,1,bob]])
    mm=cv2.warpAffine(rm_f,M,(N,N),flags=cv2.INTER_LINEAR)
    sq=sq*(1-mm[...,None])+RUN_COL*mm[...,None]
    sq=cv2.resize(np.clip(sq,0,255).astype(np.uint8),(S,S),interpolation=cv2.INTER_AREA)
    o=base_np.copy(); o[sy:sy+S,sx:sx+S]=sq; return o
if os.environ.get('TS'):
    ts=[float(v) for v in os.environ['TS'].split(',')]
    sh=Image.new('RGB',(360*len(ts),640))
    for i,t in enumerate(ts): sh.paste(Image.fromarray(frame2(t)).resize((360,640)),(360*i,0))
    sh.save('jn/asample.png')
else:
    os.makedirs('jn/framesA',exist_ok=True)
    for i in range(int(FPS*DUR)):
        Image.fromarray(frame2(i/FPS)).save(f'jn/framesA/f_{i:04d}.jpg',quality=92)
        if i%60==0: print(i,flush=True)
