import ctypes, torch, os, glob
torch.zeros(1,device='cuda')
hip=ctypes.CDLL(glob.glob(os.path.join(os.path.dirname(torch.__file__),'lib','libamdhip64.so*'))[0])
def chk(e):
    if e: raise RuntimeError(f'hip error {e}')
class Module:
    def __init__(s,path):
        s.m=ctypes.c_void_p(); s.data=open(path,'rb').read(); chk(hip.hipModuleLoadData(ctypes.byref(s.m),s.data))
    def fn(s,name):
        f=ctypes.c_void_p(); chk(hip.hipModuleGetFunction(ctypes.byref(f),s.m,name.encode())); return f
def launch(f,grid,block,args):
    vals=[ctypes.c_void_p(a.data_ptr()) if isinstance(a,torch.Tensor) else ctypes.c_int(a) for a in args]
    arr=(ctypes.c_void_p*len(vals))(*[ctypes.cast(ctypes.byref(v),ctypes.c_void_p) for v in vals])
    chk(hip.hipModuleLaunchKernel(f,grid[0],grid[1],grid[2],block,1,1,0,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),arr,None))
