import os
from sys import platform
import time
import collections
from datetime import datetime
from enum import Enum
import xml.etree.ElementTree as etxml
from PIL import Image
import numpy as np
import pybullet as p
import pybullet_data
import gym

class DroneModel(Enum):
    """Drone models enumeration class."""

    CF2X = "cf2x"   # Bitcraze Craziflie 2.0 in the X configuration
    CF2P = "cf2p"   # Bitcraze Craziflie 2.0 in the + configuration
    HB = "hb"       # Generic quadrotor (with AscTec Hummingbird inertial properties)

################################################################################

class Physics(Enum):
    """Physics implementations enumeration class."""

    PYB = "pyb"                         # Base PyBullet physics update
    DYN = "dyn"                         # Update with an explicit model of the dynamics
    PYB_GND = "pyb_gnd"                 # PyBullet physics update with ground effect
    PYB_DRAG = "pyb_drag"               # PyBullet physics update with drag
    PYB_DW = "pyb_dw"                   # PyBullet physics update with downwash
    PYB_GND_DRAG_DW = "pyb_gnd_drag_dw" # PyBullet physics update with ground effect, drag, and downwash

################################################################################

class ImageType(Enum):
    """Camera capture image type enumeration class."""

    RGB = 0     # Red, green, blue (and alpha)
    DEP = 1     # Depth
    SEG = 2     # Segmentation by object id
    BW = 3      # Black and white

################################################################################

class BaseAviary(gym.Env):
    """Base class for "drone aviary" Gym environments."""

    metadata = {'render.modes': ['human']}
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.HB,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=True,
                 record=True,
                 obstacles=True,
                 user_debug_gui=True,
                 vision_attributes=False,
                 dynamics_attributes=False
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        vision_attributes : bool, optional
            Whether to allocate the attributes needed by vision-based aviary subclasses.
        dynamics_attributes : bool, optional
            Whether to allocate the attributes needed by subclasses accepting thrust and torques inputs.

        """
        #### Constants #############################################
        self.G = 9.8
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.SIM_FREQ = freq
        self.TIMESTEP = 1./self.SIM_FREQ
        self.AGGR_PHY_STEPS = aggregate_phy_steps
        #### Parameters ############################################
        self.NUM_DRONES = num_drones
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        self.iterate=1
        self.bool_contact_history=False
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.GUI = gui
        self.RECORD = record
        self.PHYSICS = physics
        self.OBSTACLES = obstacles
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:f}, km {:f},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
            self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.MAX_THRUST = 12*self.M #the max thrust is 12 m/s^2 (collective thrust, which means without mass)
        self.MAX_RPM = np.sqrt((self.MAX_THRUST) / (4*self.KF))
        # self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        if self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.HB]:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        self.MAX_ROLL_PITCH = np.pi/18
        #### Create attributes for vision tasks ####################
        self.VISION_ATTR = vision_attributes
        if self.VISION_ATTR:
            self.IMG_RES = np.array([64, 48])
            self.IMG_FRAME_PER_SEC = 24
            self.IMG_CAPTURE_FREQ = int(self.SIM_FREQ/self.IMG_FRAME_PER_SEC)
            self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
            self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
            if self.IMG_CAPTURE_FREQ%self.AGGR_PHY_STEPS != 0:
                print("[ERROR] in BaseAviary.__init__(), aggregate_phy_steps incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
                exit()
            if self.RECORD:
                self.ONBOARD_IMG_PATH = os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/onboard-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
                os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
        #### Create attributes for dynamics control inputs #########
        self.DYNAMICS_ATTR = dynamics_attributes
        if self.DYNAMICS_ATTR:
            if self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.HB]:
                self.A = np.array([ [1, 1, 1, 1], [1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2), -1/np.sqrt(2)], [-1/np.sqrt(2), 1/np.sqrt(2), 1/np.sqrt(2), -1/np.sqrt(2)], [-1, 1, -1, 1] ])
            elif self.DRONE_MODEL == DroneModel.CF2P:
                self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
            self.INV_A = np.linalg.inv(self.A)
            self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,p.COV_ENABLE_GUI]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
                # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=2.5,
                                         cameraYaw=-90,
                                         cameraPitch=-20,
                                         cameraTargetPosition=[0, 0, 1],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            print("viewMatrix", ret[2])
            print("projectionMatrix", ret[3])
            if self.USER_DEBUG:
                #### Add input sliders to the GUI ##########################
                self.SLIDERS = -1*np.ones(4)
                for i in range(4):
                    self.SLIDERS[i] = p.addUserDebugParameter("Propeller "+str(i)+" RPM", 0, self.MAX_RPM, self.HOVER_RPM, physicsClientId=self.CLIENT)
                self.INPUT_SWITCH = p.addUserDebugParameter("Use GUI RPM", 9999, -1, 0, physicsClientId=self.CLIENT)
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
            if self.RECORD:
                #### Set the camera parameters to save frames in DIRECT mode
                self.VID_WIDTH=int(640)
                self.VID_HEIGHT=int(480)
                self.FRAME_PER_SEC = 24
                self.CAPTURE_FREQ = int(self.SIM_FREQ/self.FRAME_PER_SEC)
                self.CAM_VIEW = p.computeViewMatrixFromYawPitchRoll(distance=3,
                                                                    yaw=-30,#-30,
                                                                    pitch=-30,#-30,
                                                                    roll=0,
                                                                    cameraTargetPosition=[0, 0, 0],
                                                                    upAxisIndex=2,
                                                                    physicsClientId=self.CLIENT
                                                                    )
                self.CAM_PRO = p.computeProjectionMatrixFOV(fov=60.0,
                                                            aspect=self.VID_WIDTH/self.VID_HEIGHT,
                                                            nearVal=0.1,
                                                            farVal=1000.0
                                                            )
        #### Set initial poses #####################################
        if initial_xyzs is None:
            self.INIT_XYZS = np.vstack([np.array([float(np.random.uniform(-0.1,0.1))]), \
                                        np.array([float(np.random.uniform(-0.1,0.1))]), \
                                        np.ones(self.NUM_DRONES) *float(np.random.uniform(2.4,2.6))]).transpose().reshape(self.NUM_DRONES, 3)#z=np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)
            # self.INIT_XYZS = np.vstack([np.array([float(np.random.uniform(-0.1,0.1))]), \
            #                             np.array([float(np.random.uniform(-0.1,0.1))]), \
            #                             np.ones(self.NUM_DRONES) *float(np.random.uniform(2.5))]).transpose().reshape(self.NUM_DRONES, 3)#z=np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)
            # self.INIT_XYZS = np.vstack([np.array([0]), \
            #                             np.array([0]), \
            #                             np.ones(self.NUM_DRONES) *2.6]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs).shape == (self.NUM_DRONES,3):
            self.INIT_XYZS = initial_xyzs
        else:
            print("[ERROR] invalid initial_xyzs in BaseAviary.__init__(), try initial_xyzs.reshape(NUM_DRONES,3)")
        if initial_rpys is None:
            self.INIT_RPYS = np.zeros((self.NUM_DRONES, 3))
            # self.INIT_RPYS = np.vstack([np.array([0]), \
            #                             np.array([0]), \
            #                             np.ones(self.NUM_DRONES) *float(np.random.uniform(-3.14,3.14))]).transpose().reshape(self.NUM_DRONES, 3)
            # print('INF_INIT_RPYS',self.INIT_RPYS)
        elif np.array(initial_rpys).shape == (self.NUM_DRONES, 3):
            self.INIT_RPYS = initial_rpys
        else:
            print("[ERROR] invalid initial_rpys in BaseAviary.__init__(), try initial_rpys.reshape(NUM_DRONES,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
       
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        
        
    ################################################################################

    def reset(self):
        """Resets the environment.

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.

        """
        p.resetSimulation(physicsClientId=self.CLIENT)
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Start video recording #################################
        self._startVideoRecording()
        #### Return the initial observation ########################
        return self._computeObs()
    
    ################################################################################

    def step(self,
             action
             ):
        """Advances the environment by one simulation step.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, translated into RPMs by
            the specific implementation of `_preprocessAction()` in each subclass.

        Returns
        -------
        ndarray | dict[..]
            The step's observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        float | dict[..]
            The step's reward value(s), check the specific implementation of `_computeReward()`
            in each subclass for its format.
        bool | dict[..]
            Whether the current epoisode is over, check the specific implementation of `_computeDone()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        
        
        #save the current action
        self.current_action = action

        
       
        #### Repeat for as many as the aggregate physics steps #####
        for _ in range(self.AGGR_PHY_STEPS):
            #### Update and store the drones kinematic info for certain
            #### Between aggregate steps for certain types of update ###
            
            self._updateAndStoreKinematicInformation()
            
            clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            self._physics(clipped_action[0, :], 0)
            p.stepSimulation(physicsClientId=self.CLIENT)
            
            # if self.AGGR_PHY_STEPS > 1 and self.PHYSICS in [Physics.PYB,Physics.DYN, Physics.PYB_GND, Physics.PYB_DRAG, Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
            #     self._updateAndStoreKinematicInformation()
            #     clipped_action = np.reshape(self._preprocessAction(action), (self.NUM_DRONES, 4))
            #### Step the simulation using the desired physics update ##
            # for i in range (self.NUM_DRONES):
            #     if self.PHYSICS == Physics.PYB:
            #         self._physics(clipped_action[i, :], i)
            #     elif self.PHYSICS == Physics.DYN:
            #         self._dynamics(clipped_action[i, :], i)
            
           

            #### PyBullet computes the new state, unless Physics.DYN， Dyn will computes the state by a hand write lib ###
            # p.stepSimulation(physicsClientId=self.CLIENT)
            # if self.PHYSICS != Physics.DYN:
            #     p.stepSimulation(physicsClientId=self.CLIENT)


            #### Save the last applied action (e.g. to compute drag) ###
            self.last_clipped_action = clipped_action



            #tree branch 
            for i in range(p.getNumJoints(self.tree)):
                #disable default constraint-based motors
                p.setJointMotorControl2(self.tree, i, p.VELOCITY_CONTROL,  force=0,physicsClientId=self.CLIENT)
            
            # # print the branch joints states
            # for i in range(p.getNumJoints(self.tree)):    
            #     print('the joints',i,p.getJointState(self.tree, i))
            
            ###########    Control the branch joints   ###############
            # pd4branch=[0,0.08,1,0,100,1,15]    #pd4branch=[0,0.079,1,0,1,1,13]
            pd4branch=self.pd4branch
            # print("pd4branch",pd4branch)
            desiredPosPole=float(pd4branch[0])
            p_joint1=float(pd4branch[1])
            d_joint1=float(pd4branch[2])
            desiredPosPole2=float(pd4branch[3])
            p_joint2=float(pd4branch[4])
            d_joint2=float(pd4branch[5])
            maxForce=float(pd4branch[6])
            link = 0
            p.setJointMotorControl2(bodyUniqueId=self.tree,
                                jointIndex=link,
                                controlMode=p.POSITION_CONTROL, #PD_CONTROL,POSITION_CONTROL
                                targetPosition=desiredPosPole,
                                targetVelocity=0,
                                force=maxForce,
                                positionGain=p_joint1,
                                velocityGain=d_joint1,
                                physicsClientId=self.CLIENT)
            link = 1
            
            p.setJointMotorControl2(bodyUniqueId=self.tree,
                                jointIndex=link,
                                controlMode=p.PD_CONTROL,#PD_CONTROL,
                                targetPosition=desiredPosPole2,
                                targetVelocity=0,
                                force=maxForce,
                                positionGain=p_joint2,
                                velocityGain=d_joint2,
                                physicsClientId=self.CLIENT)

            
            
            
            
            
            ############ Collision Detection and Visualization #######################
            p.performCollisionDetection(physicsClientId=self.CLIENT)
            L=p.getContactPoints((self.DRONE_IDS[0]),physicsClientId=self.CLIENT)
            # print("drone friction", p.getDynamicsInfo(self.DRONE_IDS[0],-1)) # the drone firction coff is 0.5
            print(L)
            
            # print(p.getDynamicsInfo(self.tree,linkIndex=1,physicsClientId=self.CLIENT))
            # P=p.getContactPoints((self.tree))
            # print("rotation mat.:", np.array(p.getMatrtrixFromQuaternion(self.quat[0, :])).reshape(3, 3))
            # print("drone position:",self.pos[0, :],self.pos[0, 0],self.pos[0, 1])

            

            if len(L) !=0 :
                
                #contact point/ pybullet can not draw a point, so draw a short line here
                p.addUserDebugLine(     lineFromXYZ=L[0][6],
                                        lineToXYZ=(L[0][6][0]+0.01,L[0][6][1],L[0][6][2]),
                                        lineColorRGB=[1, 0, 0],
                                        lineWidth=100,
                                        lifeTime=0.05,
                                        physicsClientId=self.CLIENT)
                self.force_contact_world[0]=L[0][9]
                self.force_contact_world[1]=L[0][10]
                self.force_contact_world[2]=L[0][12]
                # ### Normal Force ###
                # p.addUserDebugLine(     lineFromXYZ=L[0][6],
                #                         lineToXYZ=(L[0][6][0]+L[0][7][0]*L[0][9]*0.03,L[0][6][1]+L[0][7][1]*L[0][9]*0.03,L[0][6][2]+L[0][7][2]*L[0][9]*0.03),
                #                         lineColorRGB=[0, 1, 0],
                #                         lineWidth=5,
                #                         # lifeTime=0.5,
                #                         physicsClientId=self.CLIENT)
                # ### Lateral Friction 1 ###                
                # p.addUserDebugLine(     lineFromXYZ=L[0][6],
                #                         lineToXYZ=(L[0][6][0]+L[0][11][0]*L[0][10]*0.03,L[0][6][1]+L[0][11][1]*L[0][10]*0.03,L[0][6][2]+L[0][11][2]*L[0][10]*0.03),
                #                         lineColorRGB=[1, 1, 0],
                #                         lineWidth=5,
                #                         # lifeTime=0.5,
                #                         physicsClientId=self.CLIENT
                #                                         )
                # ### Lateral Friction 2 ###  
                # p.addUserDebugLine(     lineFromXYZ=L[0][6],
                #                         lineToXYZ=(L[0][6][0]+L[0][13][0]*L[0][12]*0.03,L[0][6][1]+L[0][13][1]*L[0][12]*0.03,L[0][6][2]+L[0][13][2]*L[0][12]*0.03),
                #                         lineColorRGB=[1, 1, 1],
                #                         lineWidth=5,
                #                         # lifeTime=0.5,
                #                         physicsClientId=self.CLIENT
                #                                         )
                # ### External Force in world coordinates ###
                # p.addUserDebugLine(     lineFromXYZ=L[0][6],
                #                         lineToXYZ=(L[0][6][0]+(L[0][13][0]*L[0][12]+L[0][11][0]*L[0][10]+L[0][7][0]*L[0][9])*0.03,L[0][6][1]+(L[0][13][1]*L[0][12]+L[0][6][1]+L[0][11][1]*L[0][10]+L[0][7][1]*L[0][9])*0.03,L[0][6][2]+(L[0][13][2]*L[0][12]+L[0][11][2]*L[0][10]+L[0][7][2]*L[0][9])*0.03),
                #                         lineColorRGB=[1, 0.64, 0],
                #                         lineWidth=5,addUserDebugLine
                #                         # lifeTime=0.5,
                #                         physicsClientId=self.CLIENT
                #                                         )
                ### External Force in robot coordinates ###
                
                # p.addUserDebugLine(     lineFromXYZ=L[0][6],
                #                         lineToXYZ=(L[0][6][0]+(L[0][13][0]*L[0][12]+L[0][11][0]*L[0][10]+L[0][7][0]*L[0][9])*0.03,L[0][6][1]+(L[0][13][1]*L[0][12]+L[0][6][1]+L[0][11][1]*L[0][10]+L[0][7][1]*L[0][9])*0.03,L[0][6][2]+(L[0][13][2]*L[0][12]+L[0][11][2]*L[0][10]+L[0][7][2]*L[0][9])*0.03),
                #                         lineColorRGB=[1, 0.64, 0],
                #                         lineWidth=5,
                #                         # lifeTime=0.5,
                #                         physicsClientId=self.CLIENT
                #                                         )
                contact_start=L[0][6] #contact points on drone 
                #L13 lateralFrictionDir2  L12 lateralFriction2;
                #L11 lateralFrictionDir1  L10 lateralFriction1;
                #L7 normal forceDir          L9 normal force;
                
                forcedir=(L[0][6][0]-L[0][5][0],L[0][6][1]-L[0][5][1],L[0][6][2]-L[0][5][2])
                print('test dir',forcedir/ np.sqrt(forcedir[0]*forcedir[0]+forcedir[1]*forcedir[1]+forcedir[2]*forcedir[2]))
                contact_end=(L[0][6][0]+(L[0][13][0]*L[0][12]+L[0][11][0]*L[0][10]+L[0][7][0]*L[0][9]),L[0][6][1]+(L[0][13][1]*L[0][12]+L[0][6][1]+L[0][11][1]*L[0][10]+L[0][7][1]*L[0][9]),L[0][6][2]+(L[0][13][2]*L[0][12]+L[0][11][2]*L[0][10]+L[0][7][2]*L[0][9]))
                # print(contact_start+self.pos[0, :],contact_end)
                ###move the force from the contact point to the center of mass
                # p.addUserDebugLine(     lineFromXYZ=self.pos[0, :],
                #                         lineToXYZ= np.array(contact_end)-np.array(contact_start)+self.pos[0, :],
                #                         lineColorRGB=[1, 0.64, 0],
                #                         lineWidth=5,
                #                         # lifeTime=0.5,
                                        
                #                         physicsClientId=self.CLIENT
                #                                         )
                ### homogeneous transformation matrix & calculate the transformal##

                rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[0, :])).reshape(3, 3)

                #-np.array(contact_start)+self.pos[0, :]
                contact_r_frame=  np.dot(rot_mat.T,(np.array(contact_end)-np.array(contact_start)+self.pos[0, :]))-np.dot(rot_mat.T,self.pos[0, :]) 
                # contact_r_frame=  np.dot(rot_mat.T,(np.array(contact_end)))-np.dot(rot_mat.T,L[0][6]) 
                # print("in robot frame:",contact_r_frame,type(self.Fcontact),type(contact_r_frame))
                self.Fcontact=contact_r_frame
                
               
                ## Visualization of external Force in robot frame ##### 
                p.addUserDebugLine(                   lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=contact_r_frame*0.07,
                                                      lineColorRGB=[1, 0, 0],
                                                      lineWidth=5,
                                                      parentObjectUniqueId=self.DRONE_IDS[0],
                                                      parentLinkIndex=-1,
                                                      lifeTime=0.1,
                                                      physicsClientId=self.CLIENT
                                                      )
            else:
                self.Fcontact= np.zeros(3) ### set the contact force to zero if there if no contact##
            
             #T + F_z = R^T *mg 
            
            # rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[0, :])).reshape(3, 3)
            # # print('the thrust is',rot_mat.T*self.MAX_THRUST*(action[0]+1)/2 )
            # # print('the Fz is',self.Fcontact[2])
            # # print('RT*mg', np.dot(rot_mat.T,[0,0,self.GRAVITY]))
            # # print('F+T',self.Fcontact[2]+self.MAX_THRUST*(action[0]+1)/2)

            # print('the force applyed on the drone in world frame:',np.dot(rot_mat,[0,0,np.sum(self.applyedforce)]))
            # print('thrust world frame is',np.dot(rot_mat,[0,0,self.MAX_THRUST*(action[0]+1)/2]))
            # print('thrust in robot fame is',([0,0,self.MAX_THRUST*(action[0]+1)/2]))
            # print('F_contact in world frame is',np.dot(rot_mat,self.Fcontact))
            # print('F_contact in robot frame is',(self.Fcontact))
            # print('mg', [0,0,self.GRAVITY])
            # print('(F+T) in world fame',np.dot(rot_mat,self.Fcontact)+np.dot(rot_mat,[0,0,self.MAX_THRUST*(action[0]+1)/2]))


            # print("contact force:", self.Fcontact,self.Fcontact[1])
            # print("wall clock",time.time()-self.RESET_TIME,"sim time:",self.step_counter*self.TIMESTEP)

                
                







            
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()
        #### Prepare the return values #############################
        obs = self._computeObs()
        # print("current_action",self.current_action,"last_action",self.last_action,"diff",self.current_action-self.last_action)
        reward = self._computeReward()
        
        done = self._computeDone()
        info = self._computeInfo()
        self._saveLastAction(action)
        #### Advance the step counter ##############################
        self.step_counter = self.step_counter + (1 * self.AGGR_PHY_STEPS)
        return obs, reward, done, info
    
    ################################################################################
    
    def render(self,
               mode='human',
               close=False
               ):
        """Prints a textual output of the environment.

        Parameters
        ----------
        mode : str, optional
            Unused.
        close : bool, optional
            Unused.

        """
        if self.first_render_call and not self.GUI:
            print("[WARNING] BaseAviary.render() is implemented as text-only, re-initialize the environment using Aviary(gui=True) to use PyBullet's graphical interface")
            self.first_render_call = False
        print("\n[INFO] BaseAviary.render() ——— it {:04d}".format(self.step_counter),
              "——— wall-clock time {:.1f}s,".format(time.time()-self.RESET_TIME),
              "simulation time {:.1f}s@{:d}Hz ({:.2f}x)".format(self.step_counter*self.TIMESTEP, self.SIM_FREQ, (self.step_counter*self.TIMESTEP)/(time.time()-self.RESET_TIME)))
        for i in range (self.NUM_DRONES):
            print("[INFO] BaseAviary.render() ——— drone {:d}".format(i),
                  "——— x {:+06.2f}, y {:+06.2f}, z {:+06.2f}".format(self.pos[i, 0], self.pos[i, 1], self.pos[i, 2]),
                  "——— velocity {:+06.2f}, {:+06.2f}, {:+06.2f}".format(self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]),
                  "——— roll {:+06.2f}, pitch {:+06.2f}, yaw {:+06.2f}".format(self.rpy[i, 0]*self.RAD2DEG, self.rpy[i, 1]*self.RAD2DEG, self.rpy[i, 2]*self.RAD2DEG),
                  "——— action_0 {:+06.2f}, action_1 {:+06.2f}, action_2 {:+06.2f}, action_3 {:+06.2f}".format(self.last_action[i, 0], self.last_action[i, 1], self.last_action[i, 2], self.last_action[i, 3]))
                  
                  #"——— angular velocity {:+06.4f}, {:+06.4f}, {:+06.4f} ——— ".format(self.ang_v[i, 0], self.ang_v[i, 1], self.ang_v[i, 2]))
    
    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        if self.RECORD and self.GUI:
            p.stopStateLogging(self.VIDEO_ID, physicsClientId=self.CLIENT)
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = 0
        
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.current_action = np.zeros((self.NUM_DRONES, 4)) # the current output of RL MLP
        self.last_action = np.zeros((self.NUM_DRONES, 4)) # the history output of RL MLP#-1*np.ones((self.NUM_DRONES, 4))
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        # self.last_output_action= np.zeros((self.NUM_DRONES, 4)) # the history output of RL MLP
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rotaion_matrix=np.zeros((self.NUM_DRONES, 9))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        #### Random Initialize the drones position information ##########
        self.INIT_XYZS = np.vstack([np.array([float(np.random.uniform(-0.1,0.1))]), \
                                        np.array([float(np.random.uniform(-0.1,0.1))]), \
                                        np.ones(self.NUM_DRONES) *float(np.random.uniform(2.4,2.6))]).transpose().reshape(self.NUM_DRONES, 3)

        # self.INIT_XYZS = np.vstack([np.array([float(np.random.uniform(-0.1,0.1))]), \
        #                                 np.array([float(np.random.uniform(-0.1,0.1))]), \
        #                                 np.ones(self.NUM_DRONES) *float(2.5)]).transpose().reshape(self.NUM_DRONES, 3)
        # self.INIT_XYZS = np.vstack([np.array([float(0)]), \
        #                                 np.array([float(0)]), \
        #                                 np.ones(self.NUM_DRONES) *float(2.5)]).transpose().reshape(self.NUM_DRONES, 3)
        #### Initialize the branch friction friction coefficient ##########
        self.lateralFriction=float(np.random.uniform(0.8,0.1))
        # self.lateralFriction=0.1
        #### Initialize the drones contact force information ##########
        self.Fcontact= np.zeros(3)
        self.force_contact_world=np.zeros(3)
        self.applyedforce=np.zeros(4)

        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### reset the branch parameter
        self.pd4branch=[ 
        np.random.uniform(-0.01,0.01),##random pos value in x-axis,
        np.random.uniform(0.02,0.1),##random p value in x-axis,
        np.random.uniform(0.8,1.2),##random d value in x-axis,
        np.random.uniform(-0.05,0.05), ##random pos in z-axis
        np.random.uniform(5,1000),##random p value in z-axis
        np.random.uniform(0.5,1),##random d value in z-axis
        np.random.uniform(3,15)]##random max_force
        # self.pd4branch=[0,0.08,1,0,100,1,5]
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)

        
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)  # 1 is enable the real time simulation
        # p.setTimeStep(self.TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)
        #### Load ground plane, drone and obstacles models #########
        self.PLANE_ID = p.loadURDF("plane.urdf", physicsClientId=self.CLIENT)
        # p.createCollisionShape(p.GEOM_PLANE)
        self.DRONE_IDS = np.array([p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF,
                                              self.INIT_XYZS[i,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS[i,:]),
                                            #   flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])
        self.tree=p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/treebranch.urdf",
        
                   [0, 1, 0],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT,
                   useFixedBase=True,
                #    flags =p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT, #self collision
                   )
        #change the branch friction coefficient
        p.changeDynamics(self.tree,linkIndex=1,physicsClientId=self.CLIENT,lateralFriction=self.lateralFriction)
        
        # add the local axes to the drone, but this will slows down the GUI
        self._showDroneLocalAxes(0)

        # for i in range(self.NUM_DRONES):
        #     #### Show the frame of reference of the drone, note that ###
        #     #### It severly slows down the GUI #########################
        #     if self.GUI and self.USER_DEBUG:
        #         self._showDroneLocalAxes(i)
        #     #### Disable collisions between drones' and the ground plane
        #     #### E.g., to start a drone at [0,0,0] #####################
        #     # p.setCollisionFilterPair(bodyUniqueIdA=self.PLANE_ID, bodyUniqueIdB=self.DRONE_IDS[i], linkIndexA=-1, linkIndexB=-1, enableCollision=0, physicsClientId=self.CLIENT)
        #pidpara=self.pd4branch
        # if self.OBSTACLES:
        #     self._addObstacles()
    
    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rotaion_matrix[i]=p.getMatrixFromQuaternion(self.quat[i])
    ################################################################################
    
    
    
    def _contactdetection(self):
        print("test")




    ###############################################################################
    def _startVideoRecording(self):
        """Starts the recording of a video output.

        The format of the video output is .mp4, if GUI is True, or .png, otherwise.
        The video is saved under folder `files/videos`.

        """
        if self.RECORD and self.iterate % 500: #and self.GUI
            print('start recording ....')
            self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
                                                fileName=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+str(self.iterate)+".mp4",
                                                physicsClientId=self.CLIENT
                                                )
       
        # if self.iterate % 100 == 0:
        #     print('start recording ....')
        #     self.VIDEO_ID = p.startStateLogging(loggingType=p.STATE_LOGGING_VIDEO_MP4,
        #                                         fileName=os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+str(self.iterate)+".mp4",
        #                                         physicsClientId=self.CLIENT
        #                                         )
        # if self.RECORD and not self.GUI:
        #     self.FRAME_NUM = 0
        #     self.IMG_PATH = os.path.dirname(os.path.abspath(__file__))+"/../../files/videos/video-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+"/"
        #     os.makedirs(os.path.dirname(self.IMG_PATH), exist_ok=True)
    
    ################################################################################

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.
        
        """
        state = np.hstack([self.pos[nth_drone, :], self.rotaion_matrix[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_action[nth_drone, :],self.Fcontact])
        return state.reshape(25,)

    ################################################################################

    def _getDroneImages(self,
                        nth_drone,
                        segmentation: bool=True
                        ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.
        segmentation : bool, optional
            Whehter to compute the compute the segmentation mask.
            It affects performance.

        Returns
        -------
        ndarray 
            (h, w, 4)-shaped array of uint8's containing the RBG(A) image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the depth image captured from the n-th drone's POV.
        ndarray
            (h, w)-shaped array of uint8's containing the segmentation image captured from the n-th drone's POV.

        """
        if self.IMG_RES is None:
            print("[ERROR] in BaseAviary._getDroneImages(), remember to set self.IMG_RES to np.array([width, height])")
            exit()
        rot_mat = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(self.pos[nth_drone, :])
        DRONE_CAM_VIEW = p.computeViewMatrix(cameraEyePosition=self.pos[nth_drone, :]+np.array([0, 0, self.L]),
                                             cameraTargetPosition=target,
                                             cameraUpVector=[0, 0, 1],
                                             physicsClientId=self.CLIENT
                                             )
        DRONE_CAM_PRO =  p.computeProjectionMatrixFOV(fov=60.0,
                                                      aspect=1.0,
                                                      nearVal=self.L,
                                                      farVal=1000.0
                                                      )
        SEG_FLAG = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = p.getCameraImage(width=self.IMG_RES[0],
                                                 height=self.IMG_RES[1],
                                                 shadow=1,
                                                 viewMatrix=DRONE_CAM_VIEW,
                                                 projectionMatrix=DRONE_CAM_PRO,
                                                 flags=SEG_FLAG,
                                                 physicsClientId=self.CLIENT
                                                 )
        rgb = np.reshape(rgb, (h, w, 4))
        dep = np.reshape(dep, (h, w))
        seg = np.reshape(seg, (h, w))
        return rgb, dep, seg

    ################################################################################

    def _exportImage(self,
                     img_type: ImageType,
                     img_input,
                     path: str,
                     frame_num: int=0
                     ):
        """Returns camera captures from the n-th drone POV.

        Parameters
        ----------
        img_type : ImageType
            The image type: RGB(A), depth, segmentation, or B&W (from RGB).
        img_input : ndarray
            (h, w, 4)-shaped array of uint8's for RBG(A) or B&W images.
            (h, w)-shaped array of uint8's for depth or segmentation images.
        path : str
            Path where to save the output as PNG.
        fram_num: int, optional
            Frame number to append to the PNG's filename.

        """
        if img_type == ImageType.RGB:
            (Image.fromarray(img_input.astype('uint8'), 'RGBA')).save(path+"frame_"+str(frame_num)+".png")
        elif img_type == ImageType.DEP:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.SEG:
            temp = ((img_input-np.min(img_input)) * 255 / (np.max(img_input)-np.min(img_input))).astype('uint8')
        elif img_type == ImageType.BW:
            temp = (np.sum(img_input[:, :, 0:2], axis=2) / 3).astype('uint8')
        else:
            print("[ERROR] in BaseAviary._exportImage(), unknown ImageType")
            exit()
        if img_type != ImageType.RGB:
            (Image.fromarray(temp)).save(path+"frame_"+str(frame_num)+".png")

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix 
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                if np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j+i+1] = adjacency_mat[j+i+1, i] = 1
        return adjacency_mat
    
    ################################################################################
    
    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2)*self.KF
        self.applyedforce=forces
        print("the real force:",forces)
        torques = np.array(rpm**2)*self.KM
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):

            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )

    ################################################################################

    

    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone,:]
        quat = self.quat[nth_drone,:]
        rpy = self.rpy[nth_drone,:]
        vel = self.vel[nth_drone,:]
        rpy_rates = self.rpy_rates[nth_drone,:]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2)*self.KM
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL==DroneModel.CF2X or self.DRONE_MODEL==DroneModel.HB:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.TIMESTEP * rpy_rates_deriv
        pos = pos + self.TIMESTEP * vel
        rpy = rpy + self.TIMESTEP * rpy_rates
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          p.getQuaternionFromEuler(rpy),
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            [-1, -1, -1], # ang_vel not computed by DYN
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone,:] = rpy_rates
    
    ################################################################################

    def _normalizedActionToRPM(self,
                               action
                               ):
        """De-normalizes the [-1, 1] range to the [0, MAX_RPM] range.

        Parameters
        ----------
        action : ndarray
            (4)-shaped array of ints containing an input in the [-1, 1] range.

        Returns
        -------
        ndarray
            (4)-shaped array of ints containing RPMs for the 4 motors in the [0, MAX_RPM] range.

        """
        if np.any(np.abs(action) > 1):
            print("\n[ERROR] it", self.step_counter, "in BaseAviary._normalizedActionToRPM(), out-of-bound action")
        return np.where(action <= 0, (action+1)*self.HOVER_RPM, self.HOVER_RPM + (self.MAX_RPM - self.HOVER_RPM)*action) # Non-linear mapping: -1 -> 0, 0 -> HOVER_RPM, 1 -> MAX_RPM`

    ################################################################################


    
    ################################################################################

    def _saveLastAction(self,
                        action
                        ):
        """Stores the most recent action into attribute `self.last_action`.

        The last action can be used to compute aerodynamic effects.
        The method disambiguates between array and dict inputs 
        (for single or multi-agent aviaries, respectively).

        Parameters
        ----------
        action : ndarray | dict
            (4)-shaped array of ints (or dictionary of arrays) containing the current RPMs input.

        """
        if isinstance(action, collections.abc.Mapping):
            for k, v in action.items(): 
                res_v = np.resize(v, (1, 4)) # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
                self.last_action[int(k), :] = res_v
        else: 
            res_action = np.resize(action, (1, 4)) # Resize, possibly with repetition, to cope with different action spaces in RL subclasses
            self.last_action = np.reshape(res_action, (self.NUM_DRONES, 4))
    
    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
    
    ################################################################################

    def _addObstacles(self):#, pd4branch
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        # p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("duck_vhacd.urdf",
        #            [0, 1, 3],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT
        #            )
        # p.loadURDF("~/RL/gym-pybullet-drones/gym_pybullet_drones/assets/table2.urdf",
        #            [-.5, -.5, 5],
        #         #    p.getQuaternionFromEuler([0, 0, 0]),
        #         #    physicsClientId=self.CLIENT
        #            )
        
       
        ####################    load the tree branches      ########################################
        # urdf_path=os.path.join("/home/ziqiao/RL/gym-pybullet-drones/gym_pybullet_drones/assets/treebranch.urdf")        
        # self.tree=p.loadURDF(os.path.dirname(os.path.abspath(__file__))+"/../assets/treebranch.urdf",
        
        #            [0, 1, 0],
        #            p.getQuaternionFromEuler([0, 0, 0]),
        #            physicsClientId=self.CLIENT,
        #            useFixedBase=True,
        #         #    flags =p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT, #self collision
        #            )
        
        
        

        ################################################################################

        # p.loadURDF("sphere2.urdf",
        #            [0, 0, 4],
        #         #    p.getQuaternionFromEuler([0,0,0]),
        #            physicsClientId=self.CLIENT,
        #            useFixedBase=False,
        #            )
    

    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(os.path.dirname(os.path.abspath(__file__))+"/../assets/"+self.URDF).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = 0.22 #float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################
    
    def _computeObs(self):
        """Returns the current observation of the environment.

        Must be implemented in a subclass.

        """
        raise NotImplementedError
    
    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Must be implemented in a subclass.

        Parameters
        ----------
        action : ndarray | dict[..]
            The input action for one or more drones, to be translated into RPMs.

        """
        raise NotImplementedError

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeDone(self):
        """Computes the current done value(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Must be implemented in a subclass.

        """
        raise NotImplementedError
