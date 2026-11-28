"""Single-world CUDA physics with the existing CPU task and rendering interface.

Only reset preparation, observations and reward bookkeeping use host MuJoCo.
All time integration, collisions, constraints and motor control use MJWarp.
"""
import numpy as np
import mujoco
try:
    import warp as wp
    import mujoco_warp as mjw
except ImportError as error:
    raise ImportError('GPU physics needs the dual-reacher-gpu Conda environment; see environment-gpu.yml') from error

from simulation.dual_reacher import MAX_ROT


@wp.kernel
def motor_control(qpos:wp.array2d(dtype=float),qvel:wp.array2d(dtype=float),
                  bias:wp.array2d(dtype=float),targets:wp.array(dtype=float),
                  qadr:wp.array(dtype=int),dadr:wp.array(dtype=int),
                  flags:wp.array(dtype=int),ctrl:wp.array2d(dtype=float)):
    i=wp.tid()
    start=(i//2)*7+(i%2)*3
    vector=wp.vec3(targets[start],targets[start+1],targets[start+2])*MAX_ROT
    angle=wp.length(vector)
    desired=wp.quat_identity()
    if angle>1.e-10:desired=wp.quat_from_axis_angle(vector/angle,wp.min(angle,MAX_ROT))
    q=qadr[i];d=dadr[i]
    current=wp.quat(qpos[0,q+1],qpos[0,q+2],qpos[0,q+3],qpos[0,q])
    difference=wp.quat_inverse(current)*desired
    if difference[3]<0.:difference=-difference
    xyz=wp.vec3(difference[0],difference[1],difference[2])
    length=wp.length(xyz)
    error=2.*xyz
    if length>1.e-10:error=xyz*(2.*wp.atan2(length,difference[3])/length)
    for k in range(3):
        torque=180.*error[k]-16.*qvel[0,d+k]
        if flags[0]!=0:torque=torque+bias[0,d+k]
        ctrl[0,start+k]=wp.clamp(torque,-65.,65.)*float(flags[1])
    if i<2:
        index=6+7*i
        ctrl[0,index]=.0225*(1.+targets[index])


@wp.kernel
def accumulate_effort(forces:wp.array2d(dtype=float),grip:wp.array2d(dtype=float),
                      dofs:wp.array(dtype=int),timestep:float,total:wp.array(dtype=float)):
    torque=float(0.)
    for i in range(12):torque=torque+wp.abs(forces[0,dofs[i]])
    total[0]=total[0]+torque/12./65.*timestep
    total[1]=total[1]+(wp.abs(grip[0,6])+wp.abs(grip[0,13]))/2./200.*timestep


class WarpPhysics:
    """Exactly one CUDA world. No CPU integration fallback is permitted."""
    def __init__(self,env):
        wp.init()
        self.device=wp.get_device('cuda:0')
        if not self.device.is_cuda:raise RuntimeError('CUDA physics device unavailable')
        self.env=env;self.model=None;self.data=None;self.graph=None;self.dirty=True;self.substeps=0
        # MJWarp 0.0.2's dense IMPLICITFAST path is unstable for this 22-DOF
        # model. The sparse path keeps the same integrator and physical model.
        env.model.opt.jacobian=mujoco.mjtJacobian.mjJAC_SPARSE
        with wp.ScopedDevice(self.device):
            self.targets=wp.zeros(14,dtype=float)
            self.flags=wp.zeros(2,dtype=int)
            self.qadr=wp.array([q for q,_ in env.joints],dtype=int)
            self.dadr=wp.array([d for _,d in env.joints],dtype=int)
            self.dofs=wp.array(env.ball_dofs.astype(np.int32),dtype=int)
            self.effort=wp.zeros(2,dtype=float)

    def invalidate(self):
        self.dirty=True

    def _simulate(self):
        self.effort.zero_()
        # Refresh derived quantities on device, including after restoring state.
        mjw.forward(self.model,self.data)
        for _ in range(self.env.frame_skip):
            wp.launch(motor_control,dim=4,inputs=[self.data.qpos,self.data.qvel,self.data.qfrc_bias,
                self.targets,self.qadr,self.dadr,self.flags],outputs=[self.data.ctrl])
            mjw.step(self.model,self.data)
            wp.launch(accumulate_effort,dim=1,inputs=[self.data.qfrc_actuator,self.data.actuator_force,
                self.dofs,self.env.model.opt.timestep],outputs=[self.effort])
        mjw.forward(self.model,self.data)

    def _prepare(self):
        self.graph=None
        self.model=mjw.put_model(self.env.model)
        self.data=mjw.put_data(self.env.model,self.env.data,nworld=1,nconmax=300,njmax=1500)
        # Compile before capture. Warm-up changes only scratch GPU state, which
        # is replaced with the actual host state before recording the graph.
        self._simulate()
        self.data=mjw.put_data(self.env.model,self.env.data,nworld=1,nconmax=300,njmax=1500)
        with wp.ScopedCapture(device=self.device) as capture:self._simulate()
        self.graph=capture.graph
        self.dirty=False

    def advance(self,action):
        with wp.ScopedDevice(self.device):
            self.targets.assign(np.asarray(action,dtype=np.float32))
            self.flags.assign(np.array([self.env.gravity_compensation,self.env.motor_enabled],dtype=np.int32))
            if self.dirty:self._prepare()
            wp.capture_launch(self.graph)
            contacts=int(self.data.nacon.numpy()[0]);constraints=int(self.data.nefc.numpy()[0])
            if contacts>self.data.naconmax or constraints>self.data.njmax:
                raise RuntimeError('GPU contact/constraint buffer overflow; physics step rejected')
            # Copy the GPU-computed positions AND contacts/constraint forces.
            # Calling mj_forward here would replace those forces with CPU ones.
            mjw.get_data_into(self.env.data,self.env.model,self.data)
            # MuJoCo 3.3.7 keeps these deprecated fields separately from geom.
            # MJWarp exports geom only; task contact checks still use geom1/2.
            self.env.data.contact.geom1[:]=self.env.data.contact.geom[:,0]
            self.env.data.contact.geom2[:]=self.env.data.contact.geom[:,1]
            values=self.effort.numpy()
            self.substeps+=self.env.frame_skip
            if not np.isfinite(self.env.data.qpos).all() or not np.isfinite(self.env.data.qvel).all():
                raise RuntimeError('GPU physics produced nonfinite state')
            return float(values[0]),float(values[1])

    def close(self):
        self.graph=None;self.data=None;self.model=None
