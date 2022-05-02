import pybullet as p
import time
import pybullet_data
# from pdControllerExplicit import PDControllerExplicit


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)

sphereRadius = 0.05
colSphereId = p.createCollisionShape(p.GEOM_SPHERE, radius=sphereRadius)
colBoxId = p.createCollisionShape(p.GEOM_BOX,
                                  halfExtents=[sphereRadius, sphereRadius, sphereRadius])

mass = 1
visualShapeId = -1

link_Masses = [1]
linkCollisionShapeIndices = [colBoxId]
linkVisualShapeIndices = [-1]
linkPositions = [[0, 0, 0.11]]
linkOrientations = [[0, 0, 0, 1]]
linkInertialFramePositions = [[0, 0, 0]]
linkInertialFrameOrientations = [[0, 0, 0, 1]]
indices = [0]
jointTypes = [p.JOINT_REVOLUTE]
axis = [[0, 0, 1]]

for i in range(3):
  for j in range(3):
    for k in range(3):
      basePosition = [
          1 + i * 5 * sphereRadius, 1 + j * 5 * sphereRadius, 1 + k * 5 * sphereRadius + 1
      ]
      baseOrientation = [0, 0, 0, 1]
      if (k & 2):
        sphereUid = p.createMultiBody(mass, colSphereId, visualShapeId, basePosition,
                                      baseOrientation)
      else:
        sphereUid = p.createMultiBody(mass,
                                      colBoxId,
                                      visualShapeId,
                                      basePosition,
                                      baseOrientation,
                                      linkMasses=link_Masses,
                                      linkCollisionShapeIndices=linkCollisionShapeIndices,
                                      linkVisualShapeIndices=linkVisualShapeIndices,
                                      linkPositions=linkPositions,
                                      linkOrientations=linkOrientations,
                                      linkInertialFramePositions=linkInertialFramePositions,
                                      linkInertialFrameOrientations=linkInertialFrameOrientations,
                                      linkParentIndices=indices,
                                      linkJointTypes=jointTypes,
                                      linkJointAxis=axis)

      p.changeDynamics(sphereUid,
                       -1,
                       spinningFriction=0.001,
                       rollingFriction=0.001,
                       linearDamping=0.0)
      for joint in range(p.getNumJoints(sphereUid)):
        p.setJointMotorControl2(sphereUid, joint, p.VELOCITY_CONTROL, targetVelocity=1, force=10)

p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(1)
# load the tree for urdf model
tree=p.loadURDF("/home/ziqiao/RL/gym-pybullet-drones/gym_pybullet_drones/assets/treebranch.urdf",
        
                   [0, 0, 0],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=p.connect(p.GUI),
                   useFixedBase=True,
                   )

drone=p.loadURDF("/home/ziqiao/Downloads/bullet3-master/data/Quadrotor/quadrotor.urdf",
        
                   [0, 0, 0],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=p.connect(p.GUI),
                   #useFixedBase=True,
                   )

p.loadURDF("sphere2.urdf",
                   [0, 1, 5],
                   
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=p.connect(p.GUI),
                  
                   )

# get the "tree" joints information
# print(p.getNumJoints(tree))
# for i in range(p.getNumJoints(tree)):
#   print(p.getJointInfo(tree, i))
#   print(p.getJoinState(tree, i))


# Control of branch
# maxForce = 10
# p.setJointMotorControl2(bodyUniqueId=tree,
# jointIndex=1,
# controlMode=p.VELOCITY_CONTROL,
# targetVelocity = 0.5,
# force = maxForce)


for i in range(p.getNumJoints(tree)):
  #disable default constraint-based motors
  p.setJointMotorControl2(tree, i, p.POSITION_CONTROL, targetPosition=0, force=0)
#set the pd paremeter and the desierd pos/vel for the joints aloing the z-axis
timeStepId = p.addUserDebugParameter("timeStep", 0.001, 0.1, 0.01)
desiredPosCartId = p.addUserDebugParameter("desiredPos-Z_axis", -10, 10, 2)
desiredVelCartId = p.addUserDebugParameter("desiredVel-Z_axis", -10, 10, 0)
kpCartId = p.addUserDebugParameter("kp-Z_axis", 0, 2.5, 1)
kdCartId = p.addUserDebugParameter("kd-Z_axis", 0, 50, 1)
maxForceCartId = p.addUserDebugParameter("maxForce-Z_axis", 0, 100, 10)

textColor = [1, 1, 1]
shift = 0.05

#add a text to the branch
p.addUserDebugText("tree branch", [shift, 0, -.1],
                   textColor,
                   parentObjectUniqueId=tree,
                   parentLinkIndex=1)

#set the pd paremeter and the desierd pos/vel for the joints aloing the x-axis
desiredPosPoleId = p.addUserDebugParameter("desiredPos-X_axis", -1, 1, 0)
desiredVelPoleId = p.addUserDebugParameter("desiredVel-X_axis", -1, 1, 0)
kpPoleId = p.addUserDebugParameter("kp-X_axis", 0, 50, 1)
kdPoleId = p.addUserDebugParameter("kd-X_axis", 0, 50, 1)
maxForcePoleId = p.addUserDebugParameter("maxForce-X_axis", 0, 100, 10)
pd = p.loadPlugin("pdControlPlugin")
while p.isConnected():
  #p.getCameraImage(320,200)
  timeStep = p.readUserDebugParameter(timeStepId)
  p.setTimeStep(timeStep)

  desiredPosCart = p.readUserDebugParameter(desiredPosCartId)
  desiredVelCart = p.readUserDebugParameter(desiredVelCartId)
  kpCart = p.readUserDebugParameter(kpCartId)
  kdCart = p.readUserDebugParameter(kdCartId)
  maxForceCart = p.readUserDebugParameter(maxForceCartId)

  desiredPosPole = p.readUserDebugParameter(desiredPosPoleId)
  desiredVelPole = p.readUserDebugParameter(desiredVelPoleId)
  kpPole = p.readUserDebugParameter(kpPoleId)
  kdPole = p.readUserDebugParameter(kdPoleId)
  maxForcePole = p.readUserDebugParameter(maxForcePoleId)

  for i in range(p.getNumJoints(tree)):
    #print(p.getJointInfo(tree, i)
    print('the joints',i,p.getJointState(tree, i))

  if (pd >= 0):
    link = 0
    p.setJointMotorControl2(bodyUniqueId=tree,
                            jointIndex=link,
                            controlMode=p.POSITION_CONTROL, #PD_CONTROL,
                            targetPosition=desiredPosPole,
                            targetVelocity=desiredPosPole,
                            force=maxForceCart,
                            positionGain=kpCart,
                            velocityGain=kpCart)
    link = 1
    p.setJointMotorControl2(bodyUniqueId=tree,
                            jointIndex=link,
                            controlMode=p.PD_CONTROL,
                            targetPosition=desiredPosPole,
                            targetVelocity=desiredPosPole,
                            force=maxForcePole,
                            positionGain=kpPole,
                            velocityGain=kdPole)





while (1):
  keys = p.getKeyboardEvents()
  print(keys)

  time.sleep(0.01)
