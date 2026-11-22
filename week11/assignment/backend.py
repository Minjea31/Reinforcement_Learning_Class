"""Unmodified official OpenAI Gym 0.26.2 and lecture DeZero source paths."""
from pathlib import Path
import sys, os, warnings
ROOT=Path(__file__).resolve().parent
for folder in ['vendor/gym','vendor/deep-learning-from-scratch-3','.runtime/pygame/usr/lib/python3/dist-packages']:
 path=ROOT/folder
 if path.exists():sys.path.insert(0,str(path))
os.environ.setdefault('SDL_VIDEODRIVER','dummy');os.environ.setdefault('SDL_AUDIODRIVER','dummy');os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import numpy as np
import gym
warnings.filterwarnings('ignore',category=DeprecationWarning,module='gym.*')
def make_env(render_mode=None):
 if np.lib.NumpyVersion(np.__version__)>='2.0.0':raise RuntimeError('OpenAI Gym 0.26.2 requires the provided Python 3.10 / NumPy 1.26 environment.')
 env=gym.make('MountainCar-v0',render_mode=render_mode)
 if render_mode=='rgb_array':
  # Initialize only SDL display for an off-screen native Gym renderer.
  import pygame
  pygame.display.init()
  env.unwrapped.screen=pygame.Surface((env.unwrapped.screen_width,env.unwrapped.screen_height))
 return env
