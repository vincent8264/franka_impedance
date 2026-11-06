# Project proposal: robust grasp and lift with residual reinforcement learning

## What are we trying to do?

Use a Franka Panda to grasp and lift objects under uncertain conditions, such as
different object masses and friction coefficients. Study whether combining a
classical impedance controller with a learned reinforcement learning (RL)
residual makes grasping more robust. Compare three approaches:

1. Classical Cartesian impedance control.
2. Direct RL control.
3. Impedance control with residual RL.

## What are our baselines, and how are they lacking?

### Baseline 1: Cartesian impedance controller

The controller moves the end effector along a predefined grasp-and-lift
trajectory and uses a fixed gripper force. It is tuned for nominal conditions
and does not adapt when object properties change. Heavier or lower-friction
objects may slip or fail to lift.

### Baseline 2: Direct RL controller

Train an RL policy that directly outputs the robot's joint control actions,
such as joint torques. It can learn adaptive behavior from experience, but it
must learn basic robot control and robustness from scratch. This may require
more training and be less sample-efficient.

## What is our proposed improvement, and why might it work?

Use residual RL: add a small learned correction to the impedance controller's
command:

$$
u = u_{\mathrm{impedance}} + \Delta u_{\mathrm{RL}}.
$$

The impedance controller handles the basic grasp-and-lift motion. The RL policy
learns corrections when the nominal controller is insufficient. The residual
could adjust joint torques, lifting force, or gripper force in response to
different object properties.

This combines the structure of classical control with the adaptability of RL.
Comparing impedance control, direct RL, and impedance plus residual RL will
test whether giving RL a classical controller as a starting point improves
robustness and/or training efficiency compared with both baselines.

## Initial simulator choice

Use the pinned Isaac Lab 3.0 development checkout and its Newton backend,
described in the repository [README](../README.md). Keep the simulator, task,
object-property distributions, observations, rewards, and evaluation seeds
the same across all three controller variants.
