# CartPole Robot Supervisor Scheme 

## Summary
- What are we doing? Training an agent to learn how to balance a pole by moving forward and backward on an x-axis
- Actuators: 4 rotational motors as the wheels
- Sensor: Position sensor to read the angle of the pole angle off vertical and pole tip as well as the robot's position in the environment

## Requirement
1. Create a `venv` with: Python 3.8.x, download gym=21.0 and deepbots
2. Clone the [deepbot repo](https://github.com/aidudezzz/deepbots) because what we need is not included in the deepbots library installed.
3. Install all requirements inside that venv and make the whole thang an exported module `deepbots`, because in our controller, we read straight from there so change the system path accordingly.
4. Have Webots duh, configure it to use the venv Python, and run the simulation in Webots.

<img width="1108" height="1217" alt="Screenshot from 2026-02-25 00-05-55" src="https://github.com/user-attachments/assets/d55292aa-fa95-4b38-a478-ad3a639ad31a" />

## Links
[Tutorial](https://github.com/aidudezzz/deepbots-tutorials/blob/master/robotSupervisorSchemeTutorial/README.md )
