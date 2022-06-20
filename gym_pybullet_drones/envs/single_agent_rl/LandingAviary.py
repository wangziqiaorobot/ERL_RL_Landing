# from cv2 import exp
from calendar import c
import numpy as np
from gym import spaces
import pybullet as p

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary

class LandingAviary(BaseSingleAgentAviary):
    """Single agent RL problem: Landing."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB, #change the physics to the DYN inorder to the dynamics information
                 freq: int=240,
                 aggregate_phy_steps: int=12,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.LD
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        

    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)  ###  self._computeObs() need or not???
        diff_act= self.current_action-state[16:20]

        time=self.step_counter*self.TIMESTEP
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        ############## for the hovering task  #############
        # return  0.05*(np.exp(- 10*np.linalg.norm(np.array([0, 0, 0.5])-state[0:3])**4)-1) -0.02*np.linalg.norm(diff_act)**2-0.01*np.linalg.norm(self.current_action)**2+0.01*(np.exp(- np.linalg.norm(np.array([0, 0])-state[10:12])**4)-1)+0.01*(np.exp(- np.linalg.norm(np.array([0, 0,0])-state[13:16])**4)-1)+ 0.05

        
        
        
        ########### for the landing task ############# 
        # L_vel + W_vel + Contact_force + energy_consanpution
        
        balancingRewardCoeff=0.1#/(time+0.5)#0.001*(time);0.01
        slippageRewardCoeff=1.2*time#0.8;0.5;0.3
        contactRewardCoeff=0.01*time
        linearvelocityRewardCoeff=0.05#0.03/(time+0.5)
        angulervelocityRewardCoeff=0.003*time
        actionsmoothRewardCoeff=-0.01
        actionlimitRewardCoeff=-0.00001*time
        contactgroundRewardCoeff=-0.00001
        contactReward=time*-0.02
          
        if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
            slippageReward=-15
        else:
            slippageReward=slippageRewardCoeff* (np.exp(- np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-state[0:2])**8)-1) ##^14
        # slippageReward=slippageRewardCoeff* (np.exp(- np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-state[0:2])**4)-1)
        # if state[23]==0:
        
        #     slippageReward=slippageRewardCoeff*(-10)
        # else:

        #     slippageReward=slippageRewardCoeff*(np.exp(- np.linalg.norm(0-state[21:23])**4)-1)
        # if len(p.getContactPoints(self.tree,physicsClientId=self.CLIENT)) !=0: #if have contact
        if (state[22]) >0: #if have contact np.linalg.norm(state[22]) >0: 
            contactReward=0   #contactRewardCoeff*(np.exp(- np.linalg.norm(0.3-state[22])**4)-1) 
            self.bool_contact_history=True
        if  self.bool_contact_history==True:
            balancingRewardCoeff = balancingRewardCoeff/(time+0.01)

        # else:
        #     contactReward=time*-0.02
        #     balancingRewardCoeff=0.1#/(time+0.5)#0.001*(time);0.01

        balancingReward=balancingRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-state[7:10])**6)-1)
        linearvelocityReward=linearvelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0, 0])-state[10:13])**4)-1)
        angulervelocityReward=angulervelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,0])-state[13:16])**4)-1)
        actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2
        actionlimitReward=actionlimitRewardCoeff*np.linalg.norm(self.current_action[3])**2
        if len(L) !=0:
            contactgroundReward=-(10)
            print("fall down to the ground")
        else:
            contactgroundReward=0
        return balancingReward+slippageReward+contactReward+linearvelocityReward+angulervelocityReward+actionsmoothReward+actionlimitReward+contactgroundReward+0.0005
        
        #   self.step_counter*self.TIMESTEP
        





        # return  0.05*(np.exp(- np.linalg.norm(np.array([0, 0, 0.5])-state[0:3])**4)-1) -0.02*np.linalg.norm(diff_act)**2+0.01*(np.exp(- np.linalg.norm(np.array([0, 0])-state[10:12])**4)-1)+0.01*(np.exp(- np.linalg.norm(np.array([0, 0,0])-state[13:16])**4)-1)+ 0.005
        # return -1 *(np.exp(-(0-state[0])**2)-3+np.exp(-(0-state[1])**2)+np.exp(-(0.5-state[2])**2))
        # return  -1 * ((0-state[0])**2+(0-state[1])**2+(0.5-state[2])**2)*0.5                  #-1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2*0.01

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        ### TO DO : stop the current episode, when the dorne on ground
        
        p.performCollisionDetection(physicsClientId=self.CLIENT)
        L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC or len(L) !=0 or np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:# or ((self._getDroneStateVector(0))[2] < 0.05)  or ((self._getDroneStateVector(0))[2] > 1.5):
            self.iterate= self.iterate+1
            print("iterate",self.iterate)
        # Alternative done condition, see PR #32
        # if (self.step_counter/self.SIM_FREQ > (self.EPISODE_LEN_SEC)) or ((self._getDroneStateVector(0))[2] < 0.05):
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
    #     """Computes the current info dict(s).

    #     Unused.

    #     Returns
    #     -------
    #     dict[str, int]
    #         Dummy value.

    #     """

        # state = self._getDroneStateVector(0)  ###  self._computeObs() need or not???
        # diff_act= self.current_action-state[16:20]

        # time=self.step_counter*self.TIMESTEP
        # p.performCollisionDetection(physicsClientId=self.CLIENT)
        # L=p.getContactPoints(self.PLANE_ID,physicsClientId=self.CLIENT)
        # ############## for the hovering task  #############
        # # return  0.05*(np.exp(- 10*np.linalg.norm(np.array([0, 0, 0.5])-state[0:3])**4)-1) -0.02*np.linalg.norm(diff_act)**2-0.01*np.linalg.norm(self.current_action)**2+0.01*(np.exp(- np.linalg.norm(np.array([0, 0])-state[10:12])**4)-1)+0.01*(np.exp(- np.linalg.norm(np.array([0, 0,0])-state[13:16])**4)-1)+ 0.05

        
        
        
        # ########### for the landing task ############# 
        # # L_vel + W_vel + Contact_force + energy_consanpution
        
        # balancingRewardCoeff=0.1#/(time+0.5)#0.001*(time);0.01
        # slippageRewardCoeff=0.8*time#0.5;0.3
        # contactRewardCoeff=0.01*time
        # linearvelocityRewardCoeff=0.05#0.03/(time+0.5)
        # angulervelocityRewardCoeff=0.003*time
        # actionsmoothRewardCoeff=-0.01
        # actionlimitRewardCoeff=-0.00001*time
        # contactgroundRewardCoeff=-0.00001
        # contactReward=time*-0.02
          
        # if np.linalg.norm(self.pos[0,0]-self.INIT_XYZS[0][0])>1 or np.linalg.norm(self.pos[0,1]-self.INIT_XYZS[0][1])>1 or (self.pos[0,2]-self.INIT_XYZS[0][2])>1:
        #     slippageReward=-15
        # else:
        #     slippageReward=slippageRewardCoeff* (np.exp(- np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-state[0:2])**10)-1)
        # # slippageReward=slippageRewardCoeff* (np.exp(- np.linalg.norm(np.array(self.INIT_XYZS[0][0:2])-state[0:2])**4)-1)
        # # if state[23]==0:
        
        # #     slippageReward=slippageRewardCoeff*(-10)
        # # else:

        # #     slippageReward=slippageRewardCoeff*(np.exp(- np.linalg.norm(0-state[21:23])**4)-1)
        # # if len(p.getContactPoints(self.tree,physicsClientId=self.CLIENT)) !=0: #if have contact
        # if (state[22]) >0: #if have contact np.linalg.norm(state[22]) >0: 
        #     contactReward=0   #contactRewardCoeff*(np.exp(- np.linalg.norm(0.3-state[22])**4)-1) 
        #     self.bool_contact_history=True
        # if  self.bool_contact_history==True:
        #     balancingRewardCoeff = balancingRewardCoeff/(time+0.01)

        # # else:
        # #     contactReward=time*-0.02
        # #     balancingRewardCoeff=0.1#/(time+0.5)#0.001*(time);0.01

        # balancingReward=balancingRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,self.INIT_RPYS[0][2]])-state[7:10])**6)-1)
        # linearvelocityReward=linearvelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0, 0])-state[10:13])**4)-1)
        # angulervelocityReward=angulervelocityRewardCoeff*(np.exp(- np.linalg.norm(np.array([0, 0,0])-state[13:16])**4)-1)
        # actionsmoothReward=actionsmoothRewardCoeff*np.linalg.norm(diff_act)**2
        # actionlimitReward=actionlimitRewardCoeff*np.linalg.norm(self.current_action[3])**2
        # if len(L) !=0:
        #     contactgroundReward=-(10)
        #     print("fall down to the ground")
        # else:
        #     contactgroundReward=0
        # info=np.hstack([ balancingReward, contactReward,linearvelocityReward,angulervelocityReward,actionsmoothReward,actionlimitReward,slippageReward,contactgroundReward])
        return {"answer": 42} #{"answer": 42} #info
    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 2 
        MAX_LIN_VEL_Z = 1

        MAX_XY =1 #MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = 3 #MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi/2 # Full range
        
        MAX_F_XY=5  #max external froce in xy axis in robot frame
        MAX_F_Z=11.76 ##max external froce in z axis in robot frame
        #这些值正确吗？？

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)
        clipped_F_xy_External=np.clip(state[20:22],-MAX_F_XY, MAX_F_XY)
        clipped_F_z_External=np.clip(state[22],-MAX_F_Z, MAX_F_Z)
        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL      
        normalized_y =  state[9]  #state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]
        normalized_fxy_external= clipped_F_xy_External/MAX_F_XY
        normalized_fz_external= clipped_F_z_External/MAX_F_Z
        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20],
                                      normalized_fxy_external,
                                      normalized_fz_external
                                      ]).reshape(23,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in LandingAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
