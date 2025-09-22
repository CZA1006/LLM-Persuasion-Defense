import time
class T:
    def __enter__(self): self.t=time.time(); return self
    def __exit__(self,*_): self.dt=time.time()-self.t
