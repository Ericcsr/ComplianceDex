from .allegro_hand.allegro_hand_robot import AllegroHand
from .leap_hand.leap_hand_robot import LeapHand
from .allegro_hand.allegro_hand_config import ROBOT_CONFIG as allegro_config
from .leap_hand.leap_hand_config import ROBOT_CONFIG as leap_config

robot_configs = {"allegro":allegro_config, "leap":leap_config}