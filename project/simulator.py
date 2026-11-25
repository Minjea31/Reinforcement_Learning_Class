"""Interactive MuJoCo viewer and joint / payload controls.

Run: .venv/bin/python simulator.py --task random
Positions are randomized; both arms are controlled manually or by a future policy.
"""
import argparse
import ctypes
import re
import subprocess
import time
import tkinter as tk
from tkinter import ttk

import mujoco
import mujoco.viewer
import numpy as np

from dual_reacher import DualReacherEnv, MAX_ROT

TASK_LABELS = {'Random start / goal':'random'}


class Simulator:
    def __init__(self, mass=1., task='random', seed=0):
        self.env = DualReacherEnv(payload_mass=mass,task=task)
        self.env.reset(seed=seed)
        self.root = tk.Tk()
        self.root.title('Dual Reacher | Controls & Physics')
        self.root.geometry('610x980+8+40')
        self.root.configure(bg='#111722')
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.running = True
        self.paused = False
        self.syncing = False
        self.action = self.env.target_action.copy()
        self.variables = []
        self.status = tk.StringVar()
        self.measurements = tk.StringVar()
        self.torques = tk.StringVar()
        self.mass_label = tk.StringVar(value=f'{mass:.2f} kg / {mass*9.81:.2f} N')
        self.mass_var = tk.DoubleVar(value=mass)
        self.motors = tk.BooleanVar(value=True)
        self.compensation = tk.BooleanVar(value=True)
        self.task_var = tk.StringVar(value=next(label for label,value in TASK_LABELS.items() if value==task))
        self.seed_var = tk.IntVar(value=seed)
        self.scene_info = tk.StringVar()
        self._build_ui()
        self._sync_sliders()
        self.viewer = mujoco.viewer.launch_passive(self.env.model, self.env.data,
            key_callback=self.key_callback, show_left_ui=False, show_right_ui=False)
        self.viewer.cam.lookat[:] = [0., 0., .78]
        self.viewer.cam.distance = 2.6
        self.viewer.cam.azimuth = 110
        self.viewer.cam.elevation = -35
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        self.root.bind('<space>', lambda _: self.toggle_pause())
        self.root.bind('<Key-r>', lambda _: self.reset())
        self.root.bind('<Key-n>', lambda _: self.new_task())
        self.root.after(1000, self.position_viewer)
        self.root.after(20, self.tick)

    def _build_ui(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', background='#111722', foreground='#ecf0f6', font=('DejaVu Sans', 10))
        style.configure('TFrame', background='#111722')
        style.configure('TLabel', background='#111722')
        style.configure('TButton', background='#293549', foreground='#ecf0f6', padding=7)
        style.map('TButton', background=[('active','#3a4b66')])
        style.configure('TCheckbutton', background='#111722', foreground='#ecf0f6')
        style.configure('TLabelframe', background='#111722', bordercolor='#354057')
        style.configure('TLabelframe.Label', background='#111722', foreground='#aab9cd')
        style.configure('TScale', background='#111722', troughcolor='#293549')
        style.configure('TCombobox',fieldbackground='#293549',foreground='#ecf0f6')
        style.map('TCombobox',fieldbackground=[('readonly','#293549')],foreground=[('readonly','#ecf0f6')])
        style.configure('TSpinbox',fieldbackground='#293549',foreground='#ecf0f6')
        style.configure('TNotebook',background='#111722',borderwidth=0)
        style.configure('TNotebook.Tab',background='#202b3d',foreground='#aab9cd',padding=(12,8))
        style.map('TNotebook.Tab',background=[('selected','#35445e')],foreground=[('selected','#ecf0f6')])
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill='both', expand=True)
        ttk.Label(outer, text='DUAL REACHER', font=('DejaVu Sans',20,'bold')).pack(anchor='w')
        ttk.Label(outer, text='4 ball joints · 2 physical grippers · 14 controls', foreground='#aab9cd').pack(anchor='w', pady=(3,12))
        buttons = ttk.Frame(outer)
        buttons.pack(fill='x')
        self.pause_button = ttk.Button(buttons, text='Pause [Space]', command=self.toggle_pause)
        self.pause_button.pack(side='left', padx=(0,6))
        ttk.Button(buttons, text='Reset [R]', command=self.reset).pack(side='left', padx=(0,6))
        task_row = ttk.Frame(outer)
        task_row.pack(fill='x',pady=(12,0))
        mode = ttk.Combobox(task_row,textvariable=self.task_var,values=list(TASK_LABELS),state='readonly',width=25)
        mode.pack(side='left')
        mode.bind('<<ComboboxSelected>>',lambda _: self.reset())
        ttk.Label(task_row,text='Seed').pack(side='left',padx=(8,3))
        ttk.Spinbox(task_row,from_=0,to=999999,textvariable=self.seed_var,width=6).pack(side='left')
        ttk.Button(task_row,text='New task [N]',command=self.new_task).pack(side='right')
        ttk.Label(outer, textvariable=self.status, foreground='#a4d4c1').pack(anchor='w', pady=(10,5))
        notebook = ttk.Notebook(outer)
        notebook.pack(fill='both',expand=True,pady=(8,0))
        scene_tab = ttk.Frame(notebook,padding=10)
        manual_tab = ttk.Frame(notebook,padding=10)
        notebook.add(scene_tab,text='Scenario & physics')
        notebook.add(manual_tab,text='Manual joints (14 controls)')
        ttk.Label(scene_tab,textvariable=self.scene_info,justify='left',wraplength=520).pack(anchor='w',pady=(0,12))
        physics = ttk.LabelFrame(scene_tab, text='Physics', padding=10)
        physics.pack(fill='x', pady=5)
        ttk.Label(physics, text='Payload mass / gravity force').pack(anchor='w')
        ttk.Label(physics, textvariable=self.mass_label, font=('DejaVu Sans',13,'bold')).pack(anchor='w', pady=3)
        mass_scale = ttk.Scale(physics, from_=.05, to=5., variable=self.mass_var, command=self.mass_preview)
        mass_scale.pack(fill='x')
        mass_scale.bind('<ButtonRelease-1>', lambda _: self.change_mass())
        mass_scale.bind('<KeyRelease>', lambda _: self.change_mass())
        ttk.Checkbutton(physics, text='Arm motors enabled (disable to observe gravity)', variable=self.motors,
            command=lambda: setattr(self.env,'motor_enabled',self.motors.get())).pack(anchor='w', pady=(8,2))
        ttk.Checkbutton(physics, text='Gravity feedforward (motor torque includes it)', variable=self.compensation,
            command=lambda: setattr(self.env,'gravity_compensation',self.compensation.get())).pack(anchor='w')
        arms = ttk.Frame(manual_tab)
        arms.pack(fill='x', pady=10)
        for arm, side in enumerate(('LEFT ARM', 'RIGHT ARM')):
            panel = ttk.LabelFrame(arms, text=side, padding=8)
            panel.pack(side='left', fill='both', expand=True, padx=(0,7) if arm==0 else (0,0))
            for k, label in enumerate(('Shoulder X', 'Shoulder Y', 'Shoulder Z', 'Elbow X', 'Elbow Y', 'Elbow Z', 'Gripper open')):
                row = ttk.Frame(panel)
                row.pack(fill='x', pady=(2,0))
                ttk.Label(row, text=label).pack(side='left')
                variable = tk.DoubleVar()
                self.variables.append(variable)
                value_label = ttk.Label(row, width=7, anchor='e')
                value_label.pack(side='right')
                index = arm*7+k
                limit = np.degrees(MAX_ROT) if k<6 else 100.
                slider = ttk.Scale(panel, from_=-limit if k<6 else 0., to=limit, variable=variable,
                    command=lambda value, idx=index, lbl=value_label: self.change_joint(idx, value, lbl))
                slider.pack(fill='x', pady=(0,2))
                variable.trace_add('write', lambda *_, idx=index, lbl=value_label:
                    lbl.configure(text=f'{self.variables[idx].get():.0f}'+('°' if idx%7<6 else '%')))
        ttk.Label(manual_tab, text='Rotation-vector components, not Euler angles.', foreground='#aab9cd').pack(anchor='w')
        readings = ttk.LabelFrame(scene_tab, text='Live measurements', padding=10)
        readings.pack(fill='x', pady=10)
        ttk.Label(readings, textvariable=self.measurements, justify='left').pack(anchor='w')
        ttk.Label(readings, textvariable=self.torques, justify='left', font=('DejaVu Sans Mono',9)).pack(anchor='w', pady=(6,0))
        ttk.Label(scene_tab, text='New task: random start, goal, and obstacle.\nReset: restore the same scene using its seed.\n\nNo assigned roles, strategy selection, or scripted motion.\nUse manual joints now; connect a learned policy later.\nBlue / orange: arms   Yellow: payload   Red: obstacle\nGreen: goal   Viewer: drag to orbit, wheel to zoom.',
            foreground='#aab9cd', justify='left').pack(anchor='w', pady=5)

    def change_joint(self, index, value, label):
        if self.syncing:
            return
        self.action[index] = float(value)/np.degrees(MAX_ROT) if index%7<6 else float(value)/50.-1

    def _sync_sliders(self):
        self.syncing = True
        for i, var in enumerate(self.variables):
            var.set(float(self.action[i]*np.degrees(MAX_ROT) if i%7<6 else (self.action[i]+1)*50))
        self.syncing = False

    def mass_preview(self, value):
        mass = float(value)
        self.mass_label.set(f'{mass:.2f} kg / {mass*9.81:.2f} N')

    def change_mass(self):
        with self.viewer.lock():
            self.env.set_mass(float(self.mass_var.get()))

    def reset(self):
        with self.viewer.lock():
            self.env.reset(seed=self.seed_var.get(), options={'task':TASK_LABELS[self.task_var.get()]})
        self.action = self.env.target_action.copy()
        self._sync_sliders()

    def new_task(self):
        self.seed_var.set(self.seed_var.get()+1)
        self.reset()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.configure(text='Resume [Space]' if self.paused else 'Pause [Space]')

    def key_callback(self, key):
        # Called from the viewer thread: schedule UI work on Tk's main thread.
        if key in (32,82,78):
            callback = {32:self.toggle_pause,82:self.reset,78:self.new_task}[key]
            self.root.after(0, callback)

    def tick(self):
        if not self.running:
            return
        if not self.viewer.is_running():
            self.close()
            return
        started = time.monotonic()
        if not self.paused:
            self.env.step(self.action)
        info = self.env.info()
        self.scene_info.set(f"Random positions / no assigned strategy   |   Seed: {self.seed_var.get()}\n"
            f"Start: {np.array2string(self.env.start,precision=3)} m\n"
            f"Goal:  {np.array2string(self.env.goal,precision=3)} m\n"
            f"Payload: {self.env.model.geom_size[self.env.payload_geom,0]*2:.2f} m long / {self.env.mass:.2f} kg\n"
            f"Obstacle xyz: {np.array2string(info['obstacle_position'],precision=3)} m\n"
            f"Obstacle size: {np.array2string(2*info['obstacle_size'],precision=3)} m")
        stage = 'Manual control / policy-ready environment'
        self.status.set(('PAUSED · ' if self.paused else '')+stage+f' · t={self.env.data.time:.1f}s')
        pos = info['payload_position']
        self.measurements.set(f"Payload xyz: {pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:.3f} m\n"
            f"Goal distance: {info['goal_distance']:.3f} m   Two-pad grip L/R: {info['grasped']}\n"
            f"Released and settled: {info['success_hold_seconds']:.1f}/1.0 s\n"
            f"Obstacle contacts: {info['obstacle_contacts']}   Arm contacts: {info['arm_contacts']}\n"
            f"Placed: {info['placed']}   Object tilt: {info['tilt_degrees']:.1f} deg\n"
            f"Joint mechanical power: {info['mechanical_power']:+.2f} W (not electrical power)")
        torque = info['motor_torque'].reshape(2, 6)
        self.torques.set('Motor torques [N m] — shoulder XYZ / elbow XYZ\n'+
            'L '+ ' '.join(f'{v:+6.2f}' for v in torque[0])+'\n'+
            'R '+ ' '.join(f'{v:+6.2f}' for v in torque[1]))
        self.viewer.sync()
        delay = max(1, int((self.env.dt-(time.monotonic()-started))*1000))
        self.root.after(delay, self.tick)

    def position_viewer(self):
        """Place the classic viewer beside controls on an X11 desktop, if present."""
        try:
            result = subprocess.run(['xwininfo','-root','-tree'], capture_output=True, text=True, timeout=2)
            match = re.search(r'(0x[0-9a-f]+) "MuJoCo[^\"]*"', result.stdout)
            if match:
                lib = ctypes.CDLL('libX11.so.6')
                lib.XOpenDisplay.restype = ctypes.c_void_p
                lib.XMoveResizeWindow.argtypes = [ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.c_int,ctypes.c_uint,ctypes.c_uint]
                lib.XFlush.argtypes = [ctypes.c_void_p]
                lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
                display = lib.XOpenDisplay(None)
                if display:
                    lib.XMoveResizeWindow(display,int(match.group(1),16),700,80,1200,900)
                    lib.XFlush(display)
                    lib.XCloseDisplay(display)
            with self.viewer.lock():
                self.viewer.cam.lookat[:] = [0., 0., .78]
                self.viewer.cam.distance = 2.45
                self.viewer.cam.azimuth = 90
                self.viewer.cam.elevation = -30
        except (OSError,subprocess.SubprocessError):
            pass

    def close(self):
        self.running = False
        if hasattr(self,'viewer'):
            self.viewer.close()
        self.env.close()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mass',type=float,default=1.,help='Payload mass in kg, 0.05–5')
    parser.add_argument('--task',choices=list(TASK_LABELS.values()),default='random',
                        help='Randomize positions without assigning an arm or strategy (default)')
    parser.add_argument('--seed',type=int,default=0)
    args = parser.parse_args()
    Simulator(args.mass,args.task,args.seed).run()
