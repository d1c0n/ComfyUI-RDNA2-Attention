import sys,time,ctypes
sys.path.insert(0,'.')
import torch, torch.nn.functional as F
from hipload import *
def packA(a,W):
    M,K=a.shape; rows=W*64; Mp=(M+rows-1)//rows*rows
    x=torch.zeros(Mp,K,device='cuda',dtype=torch.int8); x[:M]=a
    return x.view(torch.int32).reshape(Mp//64,2,32,K//64,4,4).permute(0,3,1,4,2,5).contiguous()
def packB(b):
    N,K=b.shape
    p=b.contiguous().view(torch.int32).reshape(N//64,16,4,K//64,2,8).permute(0,3,1,4,2,5).contiguous().flatten()
    return torch.cat([p,torch.zeros(1024,device='cuda',dtype=torch.int32)])
def bench(fn,n=15):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s=time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/n*1e3
def launch_gemm(f,W,Ap,Bp,asc,C,M,N,K,ws):
    vals=[ctypes.c_void_p(Ap.data_ptr()),ctypes.c_void_p(Bp.data_ptr()),ctypes.c_void_p(asc.data_ptr()),ctypes.c_void_p(C.data_ptr()),ctypes.c_int(M),ctypes.c_int(N),ctypes.c_int(K//64),ctypes.c_float(ws)]
    arr=(ctypes.c_void_p*len(vals))(*[ctypes.cast(ctypes.byref(v),ctypes.c_void_p) for v in vals])
    rows=W*64
    chk(hip.hipModuleLaunchKernel(f,(M+rows-1)//rows,N//64,1,W*32,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),arr,None))
for path,W in [(s.split(':')[0],int(s.split(':')[1])) for s in sys.argv[1:]]:
    f=Module(path).fn('gemm')
    print('==',path,'waves',W)
    for M,K,N in ((4056,2048,8192),(4056,8192,2048),(4056,2048,2048),(3014,3072,16384)):
        torch.manual_seed(1)
        a=torch.randint(-127,128,(M,K),device='cuda',dtype=torch.int8); b=torch.randint(-127,128,(N,K),device='cuda',dtype=torch.int8)
        asc=torch.rand(M,device='cuda')*0.01+0.001; ws=0.0005; C=torch.empty(M,N,device='cuda',dtype=torch.float16)
        Ap,Bp=packA(a,W),packB(b)
        launch_gemm(f,W,Ap,Bp,asc,C,M,N,K,ws); torch.cuda.synchronize()
        rows=torch.cat([torch.arange(0,200,device='cuda'),torch.arange(M-200,M,device='cuda')])
        r=(a[rows].double()@b.double().t())*asc[rows,None].double()*ws
        e=((C[rows].double()-r).abs().max()/r.abs().max()).item()
        t=bench(lambda:launch_gemm(f,W,Ap,Bp,asc,C,M,N,K,ws))
        xf=torch.randn(M,K,device='cuda',dtype=torch.float16); wf=torch.randn(N,K,device='cuda',dtype=torch.float16); wt=wf.t().contiguous()
        tf=bench(lambda:torch.mm(xf,wt))
        print(f"{str((M,K,N)):20s} int8 {t:6.2f} ms {2*M*K*N/t/1e9:6.1f} TOPS | fp16 NN {tf:6.2f} ms {2*M*K*N/tf/1e9:5.1f} TF | speedup {tf/t:5.2f}x  relerr {e:.5f}",flush=True)
