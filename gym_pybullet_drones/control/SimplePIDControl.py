import math
import numpy as np
import pybullet as p

from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.envs.BaseAviary import DroneModel, BaseAviary
from gym_pybullet_drones.utils.utils import nnlsRPM

class SimplePIDControl(BaseControl):
    """Generic PID control class without yaw control.

    Based on https://github.com/prfraanje/quadcopter_sim.

    """

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel,
                 g: float=9.8
                 ):
        """Common control classes __init__ method.

        Parameters
        ----------
        drone_model : DroneModel
            The type of drone to control (detailed in an .urdf file in folder `assets`).
        g : float, optional
            The gravitational acceleration in m/s^2.

        """
        super().__init__(drone_model=drone_model, g=g)
        if self.DRONE_MODEL != DroneModel.HB:
            print("[ERROR] in SimplePIDControl.__init__(), SimplePIDControl requires DroneModel.HB")
            exit()
        #PID parameter of position controller
        self.P_COEFF_FOR = np.array([.065, .065, .65])
        self.I_COEFF_FOR = np.array([.000, .000, .000])
        self.D_COEFF_FOR = np.array([5.0, 5.0, 1.5])#np.array([.3, .3, .4])
        
        #PD parameter of attitude controller
        self.P_COEFF_TOR =  np.array([0.15, 0.15, .08]) #np.array([0.3, 0.3, .08])#np.array([.9, .3, .05])
        #self.I_COEFF_TOR = np.array([.0001, .0001, .0001])
        self.D_COEFF_TOR =  np.array([.05, .05, .3]) #np.array([.02, .02, .3])


        self.MAX_ROLL_PITCH = np.pi/6
        self.M=1.2
        self.L = self._getURDFParameter('arm')
        self.THRUST2WEIGHT_RATIO = self._getURDFParameter('thrust2weight')
        # self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = 12*self.M #the max thrust is 12 m/s^2 (collective thrust, which means without mass)
        self.MAX_RPM = np.sqrt((self.MAX_THRUST) / (4*self.KF))
        # self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.A = np.array([ [1, 1, 1, 1], [0, 1, 0, -1], [-1, 0, 1, 0], [-1, 1, -1, 1] ])
        self.INV_A = np.linalg.inv(self.A)
        self.B_COEFF = np.array([1/self.KF, 1/(self.KF*self.L), 1/(self.KF*self.L), 1/self.KM])
        self.reset()
       
    ################################################################################

    def reset(self):
        """Resets the control classes.

        The previous step's and integral errors for both position and attitude are set to zero.

        """
        super().reset()
        #### Initialized PID control variables #####################
        self.last_pos_e = np.zeros(3)
        self.integral_pos_e = np.zeros(3)
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
    
    ################################################################################

    def computeControl(self,
                       control_timestep,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_rpy=np.zeros(3),
                       target_vel=np.zeros(3),
                       target_rpy_rates=np.zeros(3)
                       ):
        """Computes the PID control action (as RPMs) for a single drone.

        This methods sequentially calls `_simplePIDPositionControl()` and `_simplePIDAttitudeControl()`.
        Parameters `cur_ang_vel`, `target_rpy`, `target_vel`, and `target_rpy_rates` are unused.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        cur_vel : ndarray
            (3,1)-shaped array of floats containing the current velocity.
        cur_ang_vel : ndarray
            (3,1)-shaped array of floats containing the current angular velocity.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.
        target_rpy : ndarray, optional
            (3,1)-shaped array of floats containing the desired orientation as roll, pitch, yaw.
        target_vel : ndarray, optional
            (3,1)-shaped array of floats containing the desired velocity.
        target_rpy_rates : ndarray, optional
            (3,1)-shaped array of floats containing the the desired roll, pitch, and yaw rates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.
        ndarray
            (3,1)-shaped array of floats containing the current XYZ position error.
        float
            The current yaw error.

        """
        self.control_counter += 1
        if target_rpy[2]!=0:
            print("\n[WARNING] ctrl it", self.control_counter, "in SimplePIDControl.computeControl(), desired yaw={:.0f}deg but locked to 0. for DroneModel.HB".format(target_rpy[2]*(180/np.pi)))
        thrust, computed_target_rpy, pos_e = self._simplePIDPositionControl(control_timestep,
                                                                            cur_pos,
                                                                            cur_quat,
                                                                            target_pos
                                                                            )
        # rpm = self._simplePIDAttitudeControl(control_timestep,
        #                                      thrust,
        #                                      cur_quat,
        #                                      computed_target_rpy
        #                                      )
        target_torques,rpm = self._simplePIDAttitudeControl(control_timestep,
                                             thrust,
                                             cur_quat,
                                             computed_target_rpy
                                             )
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        return rpm, pos_e,  computed_target_rpy ,cur_rpy, target_torques #

    ################################################################################

    def _simplePIDPositionControl(self,
                                  control_timestep,
                                  cur_pos,
                                  cur_quat,
                                  target_pos
                                  ):
        """Simple PID position control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_pos : ndarray
            (3,1)-shaped array of floats containing the current position.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_pos : ndarray
            (3,1)-shaped array of floats containing the desired position.

        Returns
        -------
        float
            The target thrust along the drone z-axis.
        ndarray
            (3,1)-shaped array of floats containing the target roll, pitch, and yaw.
        float
            The current position error.

        """
        pos_e = target_pos - np.array(cur_pos).reshape(3)
        d_pos_e = (pos_e - self.last_pos_e) / control_timestep
        self.last_pos_e = pos_e
        self.integral_pos_e = self.integral_pos_e + pos_e*control_timestep
        #### PID target thrust #####################################
        target_force = np.array([0, 0, self.GRAVITY]) \
                       + np.multiply(self.P_COEFF_FOR, pos_e) \
                       + np.multiply(self.I_COEFF_FOR, self.integral_pos_e) \
                       + np.multiply(self.D_COEFF_FOR, d_pos_e)
        target_rpy = np.zeros(3)
        sign_z =  np.sign(target_force[2])
        if sign_z == 0:
            sign_z = 1
        #### Target rotation #######################################
        target_rpy[0] = np.arcsin(-sign_z*target_force[1] / np.linalg.norm(target_force))
        target_rpy[1] = np.arctan2(sign_z*target_force[0], sign_z*target_force[2])
        target_rpy[2] = math.pi/2
        target_rpy[0] = np.clip(target_rpy[0], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        target_rpy[1] = np.clip(target_rpy[1], -self.MAX_ROLL_PITCH, self.MAX_ROLL_PITCH)
        cur_rotation = np.array(p.getMatrixFromQuaternion(cur_quat)).reshape(3, 3)
        thrust = np.dot(cur_rotation, target_force)
        return thrust[2], target_rpy, pos_e

    ################################################################################

    def _simplePIDAttitudeControl(self,
                                  control_timestep,
                                  thrust,
                                  cur_quat,
                                  target_rpy
                                  ):
        """Simple PID attitude control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_quat : ndarray
            (4,1)-shaped array of floats containing the current orientation as a quaternion.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the computed the target roll, pitch, and yaw.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        cur_rpy = p.getEulerFromQuaternion(cur_quat)
        # print('cur_rpy:',cur_rpy)
        # print('target_rpy:',target_rpy)
        rpy_e = target_rpy - np.array(cur_rpy).reshape(3,)
        if rpy_e[2] > np.pi:
            rpy_e[2] = rpy_e[2] - 2*np.pi
        if rpy_e[2] < -np.pi:
            rpy_e[2] = rpy_e[2] + 2*np.pi
        d_rpy_e = (rpy_e - self.last_rpy_e) / control_timestep
        self.last_rpy_e = rpy_e
        self.integral_rpy_e = self.integral_rpy_e + rpy_e*control_timestep
        
        #### PID target torques ####################################
        target_torques = np.multiply(self.P_COEFF_TOR, rpy_e) \
                         + np.multiply(self.D_COEFF_TOR, d_rpy_e)
                        #   + np.multiply(self.I_COEFF_TOR, self.integral_rpy_e) \
        #### add the torque Constraints #######
        # print('MAX_XY_TORQUE:',self.MAX_XY_TORQUE)
        # print('MAX_Z_TORQUE:',self.MAX_Z_TORQUE)
        # print('thrust:',thrust)
        target_torques[0]= np.clip(target_torques[0], -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE)
        target_torques[1]= np.clip(target_torques[1], -self.MAX_XY_TORQUE, self.MAX_XY_TORQUE)
        target_torques[2]= np.clip(target_torques[2], -self.MAX_Z_TORQUE, self.MAX_Z_TORQUE)
        
        return target_torques,nnlsRPM(thrust=thrust,
                       x_torque=target_torques[0],
                       y_torque=target_torques[1],
                       z_torque=target_torques[2],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )
    def _simplePIDBodyratesControl(self,
                                  control_timestep,
                                  thrust,
                                  cur_bodyrates,
                                  cur_bodytorque
                                  #target_bodyrates,
                                  #J,
                                  ):
        """Simple PID attitude control (with yaw fixed to 0).

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        thrust : float
            The target thrust along the drone z-axis.
        cur_bodyrates : ndarray
            (3,1)-shaped array of floats containing the current bodyrates.
        cur_bodytorque : ndarray
            (3,1)-shaped array of floats containing the current bodytorque.
        target_bodyrates : ndarray
            (3,1)-shaped array of floats containing the desired target_bodyrates.

        Returns
        -------
        ndarray
            (4,1)-shaped array of integers containing the RPMs to apply to each of the 4 motors.

        """
        # moment of inertia of the quadrotor
        J=np.array(J).reshape(3,3)
        # desired body rates;
        target_bodyrates=np.array(target_bodyrates).reshape(3,)
        # current body rates;
        cur_bodyrates=np.array(cur_bodyrates).reshape(3,)
        # cur_bodytorque;
        cur_bodytorque=np.array(cur_bodytorque).reshape(3,)
        # estimated Derivative of  desired body rates;
        d_target_bodyrates = (target_bodyrates- self.last_rpy_e) / control_timestep
        # last target bodyrates
        self.last_rpy_e = target_bodyrates
        # comput the reference body torque
        ref_bodytorques=np.cross(target_bodyrates, np.dot(J, target_bodyrates)) + np.dot(J, d_target_bodyrates)
        
        
        #### PID target torques ####################################
        target_torques = np.multiply(self.P_COEFF_TOR, target_bodyrates-cur_bodyrates) \
                         + np.multiply(self.D_COEFF_TOR, ref_bodytorques-cur_bodytorque) \
                         + np.cross(cur_bodyrates, np.dot(J, cur_bodyrates)) \
                         + np.dot(J, d_target_bodyrates)
        
        return nnlsRPM(thrust=thrust,
                       x_torque=target_torques[0],
                       y_torque=target_torques[1],
                       z_torque=target_torques[2],
                       counter=self.control_counter,
                       max_thrust=self.MAX_THRUST,
                       max_xy_torque=self.MAX_XY_TORQUE,
                       max_z_torque=self.MAX_Z_TORQUE,
                       a=self.A,
                       inv_a=self.INV_A,
                       b_coeff=self.B_COEFF,
                       gui=True
                       )
 