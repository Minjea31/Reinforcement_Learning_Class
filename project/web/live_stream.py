"""Authenticated local MJPEG source, reachable remotely only through the Site proxy."""
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import threading
import time
from urllib.parse import urlparse


class FrameHub:
    def __init__(self):
        self.condition=threading.Condition();self.jpeg=None;self.sequence=0;self.readers=0;self.closed=False

    def publish(self,jpeg):
        with self.condition:
            self.jpeg=jpeg;self.sequence+=1;self.condition.notify_all()

    def close(self):
        with self.condition:self.closed=True;self.condition.notify_all()


def start_stream(port,token,hub):
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_GET(self):
            if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_response(401);self.send_header('Content-Length','0');self.end_headers();return
            path=urlparse(self.path).path
            if path=='/health':
                import json
                with hub.condition:body=json.dumps(dict(sequence=hub.sequence,readers=hub.readers)).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            if path!='/stream':
                self.send_response(404);self.send_header('Content-Length','0');self.end_headers();return
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control','no-store, no-transform')
            self.send_header('X-Accel-Buffering','no')
            self.send_header('Connection','close');self.end_headers()
            self.connection.settimeout(5.)
            sequence=-1
            with hub.condition:hub.readers+=1
            try:
                while True:
                    with hub.condition:
                        if not hub.closed and hub.sequence==sequence:hub.condition.wait(timeout=1.)
                        if hub.closed:break
                        jpeg=hub.jpeg;sequence=hub.sequence
                    if jpeg is None:continue
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(jpeg)).encode()+b'\r\n\r\n'+jpeg+b'\r\n');self.wfile.flush()
            except (BrokenPipeError,ConnectionError,TimeoutError,OSError):pass
            finally:
                with hub.condition:hub.readers-=1
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    return server
