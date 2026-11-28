"""Capture only an existing MuJoCo client window, including when occluded."""
import ctypes as C
from ctypes.util import find_library
from pathlib import Path
import re
import subprocess

from PIL import Image


def is_descendant(pid,parent):
    for _ in range(16):
        if pid==parent:return True
        try:
            status=Path(f'/proc/{pid}/status').read_text()
            pid=int(re.search(r'^PPid:\s+(\d+)',status,re.M)[1])
        except (OSError,TypeError):return False
        if pid<=1:return False
    return False


def find_viewer(training_pid):
    result=subprocess.run(['xwininfo','-root','-tree'],capture_output=True,text=True,check=True)
    for line in result.stdout.splitlines():
        match=re.search(r'(0x[0-9a-fA-F]+) "MuJoCo : Dual spherical-joint Reacher"',line)
        if not match:continue
        window=int(match[1],16)
        props=subprocess.run(['xprop','-id',hex(window),'_NET_WM_PID'],capture_output=True,text=True,check=True)
        owner=re.search(r'=\s*(\d+)',props.stdout)
        if owner and is_descendant(int(owner[1]),training_pid):return window,int(owner[1])
    raise RuntimeError('No local MuJoCo viewer belongs to this training process')


class XImage(C.Structure):
    _fields_=[('width',C.c_int),('height',C.c_int),('xoffset',C.c_int),('format',C.c_int),
              ('data',C.c_void_p),('byte_order',C.c_int),('bitmap_unit',C.c_int),
              ('bitmap_bit_order',C.c_int),('bitmap_pad',C.c_int),('depth',C.c_int),
              ('bytes_per_line',C.c_int),('bits_per_pixel',C.c_int),
              ('red_mask',C.c_ulong),('green_mask',C.c_ulong),('blue_mask',C.c_ulong)]


def scene_only(image):
    """Remove native side panels and the top telemetry overlay, not the scene.

    Read actual window pixels; never replay a policy or invent intermediate
    frames. Dark-gray UI panels are detected independently on either side.
    """
    width,height=image.size
    left=0;right=width;y=max(0,height-8)
    for side in (0,1):
        edge=0 if side==0 else width-1
        color=image.getpixel((edge,y))
        if max(color)-min(color)>2 or not 30<=color[0]<=110:continue
        if any(image.getpixel((edge,row))!=color for row in (height//2,height*3//4)):continue
        count=0
        while count<width//3 and image.getpixel((count if side==0 else width-1-count,y))==color:count+=1
        if not 20<=count<width//3:continue
        if side==0:left=count
        else:right=width-count
    top=min(160,max(96,round(height*.14)))
    if right-left<100 or height-top<100:raise RuntimeError('MuJoCo scene viewport is too small')
    return image.crop((left,top,right,height))


class WindowCapture:
    def __init__(self,window,owner_pid,training_pid):
        if not is_descendant(owner_pid,training_pid):raise ValueError('Window is outside the training process tree')
        self.window=window;self.owner_pid=owner_pid;self.training_pid=training_pid
        self.x=C.CDLL(find_library('X11'));self.composite=C.CDLL(find_library('Xcomposite'))
        self.x.XOpenDisplay.argtypes=[C.c_char_p];self.x.XOpenDisplay.restype=C.c_void_p
        self.x.XCloseDisplay.argtypes=[C.c_void_p]
        self.x.XGetGeometry.argtypes=[C.c_void_p,C.c_ulong,C.POINTER(C.c_ulong),C.POINTER(C.c_int),C.POINTER(C.c_int),
                                    C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.POINTER(C.c_uint),C.POINTER(C.c_uint)]
        self.x.XGetGeometry.restype=C.c_int
        self.x.XGetImage.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_ulong,C.c_int]
        self.x.XGetImage.restype=C.POINTER(XImage)
        self.x.XDestroyImage.argtypes=[C.POINTER(XImage)]
        self.x.XFreePixmap.argtypes=[C.c_void_p,C.c_ulong]
        self.x.XSync.argtypes=[C.c_void_p,C.c_int]
        self.composite.XCompositeNameWindowPixmap.argtypes=[C.c_void_p,C.c_ulong]
        self.composite.XCompositeNameWindowPixmap.restype=C.c_ulong
        self.composite.XCompositeRedirectWindow.argtypes=[C.c_void_p,C.c_ulong,C.c_int]
        self.composite.XCompositeUnredirectWindow.argtypes=[C.c_void_p,C.c_ulong,C.c_int]
        self.error=False
        self.callback=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)(self._error)
        self.x.XSetErrorHandler.argtypes=[type(self.callback)];self.x.XSetErrorHandler(self.callback)
        self.display=self.x.XOpenDisplay(None)
        if not self.display:raise RuntimeError('Cannot open the local X11 display')
        self.composite.XCompositeRedirectWindow(self.display,self.window,0)
        self.x.XSync(self.display,0)
        if self.error:
            self.x.XCloseDisplay(self.display);self.display=None
            raise RuntimeError('Could not redirect the training window for isolated capture')

    def _error(self,*_):
        self.error=True
        return 0

    def capture(self):
        if not is_descendant(self.owner_pid,self.training_pid):raise RuntimeError('The training viewer has exited')
        self.error=False
        root=C.c_ulong();x=C.c_int();y=C.c_int();w=C.c_uint();h=C.c_uint();border=C.c_uint();depth=C.c_uint()
        ok=self.x.XGetGeometry(self.display,self.window,C.byref(root),C.byref(x),C.byref(y),
                              C.byref(w),C.byref(h),C.byref(border),C.byref(depth))
        if not ok or self.error or w.value<1 or h.value<1:raise RuntimeError('Training window is unavailable')
        pixmap=self.composite.XCompositeNameWindowPixmap(self.display,self.window)
        self.x.XSync(self.display,0)
        if self.error or not pixmap:raise RuntimeError('Window-only capture requires a mapped composited X11 window')
        pointer=None
        try:
            pointer=self.x.XGetImage(self.display,pixmap,0,0,w.value,h.value,C.c_ulong(-1).value,2)
            if not pointer or self.error:raise RuntimeError('Could not read the MuJoCo window pixmap')
            image=pointer.contents
            # XGetImage on a 24-bit pixmap may leave visual color masks zero.
            masks=(image.red_mask,image.green_mask,image.blue_mask)
            if image.bits_per_pixel!=32 or image.byte_order!=0 or image.depth!=24 or masks not in ((0xff0000,0xff00,0xff),(0,0,0)):
                raise RuntimeError('Unsupported window pixel format')
            raw=C.string_at(image.data,image.bytes_per_line*image.height)
            return Image.frombytes('RGB',(image.width,image.height),raw,'raw','BGRX',image.bytes_per_line,1)
        finally:
            if pointer:self.x.XDestroyImage(pointer)
            self.x.XFreePixmap(self.display,pixmap)

    def close(self):
        if self.display:
            self.composite.XCompositeUnredirectWindow(self.display,self.window,0)
            self.x.XCloseDisplay(self.display);self.display=None
