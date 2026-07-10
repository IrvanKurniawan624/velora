import tqdm

# Patch tqdm to prevent "WinError 6: The handle is invalid" on Windows when running under pytest
orig_init = tqdm.tqdm.__init__

def safe_init(self, *args, **kwargs):
    kwargs["disable"] = True
    orig_init(self, *args, **kwargs)

tqdm.tqdm.__init__ = safe_init
