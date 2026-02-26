import sys
sys.path.append("/home/khanhdo05/dev/AI/deepbots")

from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition

from gym.spaces import Box, Discrete
import numpy as np

class CartpoleRobot(RobotSupervisorEnv):
    """
    Contains required methods to run an RL training loop and get information from 
        the simulation and the robot sensors, but also control the robot.
    """
    def __init__(self):
        super().__init__()
        
        # input size and values of the agent
        # |  num  |       observation      |   min    |   max   |
        # |   0   |  Cart position x axis  |   -0.4   | 0.4     |
        # |   1   |  Cart Velocity         |   -inf   | inf     |
        # |   2   |  Pole Angle            | -1.3 rad | 1.3 rad |
        # |   3   |  Pole Velocity at Tip  |   -inf   |  inf    |
        self.observation_space = Box(low=np.array([-0.4, -np.inf, -1.3, -np.inf]),
                                     high=np.array([0.4, np.inf, 1.3, np.inf]),
                                     dtype=np.float64)
        # outputs of the agent
        # 2 is: 1 for forward movement, 1 for backward movement
        self.action_space = Discrete(2)
        
        # grab robot ref from supervisor
        self.robot = self.getSelf()
        
        self.position_sensor = self.getDevice("polePosSensor")
        self.position_sensor.enable(self.timestep)
        self.pole_endpoint = self.getFromDef("POLE_ENDPOINT")
        
        self.wheels = []
        for wheel_name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.getDevice(wheel_name)
            wheel.setPosition(float('inf'))  # set starting position
            wheel.setVelocity(0.0)  # zero out starting velocity
            self.wheels.append(wheel)
            
        self.steps_per_episode = 200  # Max number of steps per episode
        self.episode_score = 0  # Score accumulated during an episode
        self.episode_score_list = []  # A list to save all the episode scores, used to check if task is solved
        
    def get_observations(self):
        """
        Create the observation for our agent in each step
        """
        # position on x-axis
        cart_pos = normalize_to_range(self.robot.getPosition()[0], -0.4, 0.4, -1.0, 1.0)
        # linear velocity on x-axis
        cart_velocity = normalize_to_range(self.robot.getVelocity()[0], -0.2, 0.2, -1.0, 1.0, clip=True)
        # pole angle off vertical
        pole_angle = normalize_to_range(self.position_sensor.getValue(), -0.23, 0.23, -1.0, 1.0, clip=True)
        # angular velocity y of endpoint
        endpoint_velocity = normalize_to_range(self.pole_endpoint.getVelocity()[4], -1.5, 1.5, -1.0, 1.0, clip=True)
        
        return [cart_pos, cart_velocity, pole_angle, endpoint_velocity]
        
    def get_default_observation(self):
        """
        Used internally by the reset method
        """
        return [0.0 for _ in range(self.observation_space.shape[0])]
        
    def get_reward(self, action):
        """
        Return the reward for agent for each step
        
        For simplicity, agent get +1 reward for each step it manages to
            keep the pole from falling.
        """
        return 1
        
    def is_done(self):
        """
        Look for the episode done condition
        Returns a boolean
        """
        if self.episode_score > 195.0:
            return True
            
        pole_angle = round(self.position_sensor.getValue(), 2)
        if abs(pole_angle) > 0.261799388:  # more than 15 degrees off vertical (defined in radians)
            return True
        
        cart_position = round(self.robot.getPosition()[0], 2)  # Position on x-axis
        if abs(cart_position) > 0.39:
            return True

        return False
        
    def solved(self):
        """
        Look for a condition that shows the agent is fuly
            trained and able to solve the problem adequately
            
        Depends on the agent completing consecutive episodes successfully, consistently
        Method: take average episode score of last 100 episodes and check if it's > 195
        """
        if len(self.episode_score_list) > 100:  # Over 100 trials thus far
            if np.mean(self.episode_score_list[-100:]) > 195.0:  # Last 100 episodes' scores average value
                return True
        return False
        
    def apply_action(self, action):
        """
        Take the action provided by the agent and apply it to
            the robot by setting its motors' speeds
            
        Agent outputs 0 or 1 meaning backward or forward
        """
        action = int(action[0])
        if action == 0:
            motor_speed = 5.0
        else:
            motor_speed = -5.0
            
        for i in range(len(self.wheels)):
            self.wheels[i].setPosition(float('inf'))
            self.wheels[i].setVelocity(motor_speed)
        
    def get_info(self):
        return None
        
    def render(self, mode='human'):
        pass
            
             
env = CartpoleRobot()
agent = PPOAgent(number_of_inputs = env.observation_space.shape[0], number_of_actor_outputs = env.action_space.n)
solved = False
episode_count = 0
episode_limit = 2000

while not solved and episode_count < episode_limit:
    observation = env.reset()
    env.episode_score = 0
    
    # runs for the course of an episode
    for step in range(env.steps_per_episode):
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        
        # step the supervisor to get the current selected_action's reward, the new observation and whether we reached
        # the done condition
        new_observation, reward, done, info = env.step([selected_action])

        # save the current state transition in agent's memory
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)
        
        if done:
            # save the episode's score
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size = step + 1)
            solved = env.solved()
            break

        env.episode_score += reward  # accumulate episode reward
        observation = new_observation
    
    print("Episode #", episode_count, "score:", env.episode_score)
    episode_count += 1  # increment episode counter
        
if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0
while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()    
        