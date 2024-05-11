import pdb
import ray
import gym
from gym.spaces import Discrete, MultiDiscrete, Dict, Box, MultiBinary
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from PIL import Image, ImageDraw, ImageFont


class MultiAgentArena_v1d_5_RandA2(MultiAgentEnv):
    def __init__(self, config=None):
        config = config or {}
        # Dimensions of the grid.
        self.width = config.get("width", 10)
        self.height = config.get("height", 10)

        # Size of vision field
        self.vision_field = 7

        # End an episode after this many timesteps.
        self.timestep_limit = config.get("ts", 100)

        self.observation_space = Box(low=0, high=1, shape=(self.width * self.height * 2, 1), dtype=int)
        # self.observation_space = MultiDiscrete([self.width * self.height,
        #                                         self.width * self.height])
        # 0=up, 1=right, 2=down, 3=left.
        self.action_space = Discrete(4)

        # Reset env.
        self.reset()

    def reset(self):
        """Returns initial observation of next(!) episode."""
        # Row-major coords.
        self.agent1_pos = [np.random.randint(self.height - 1), np.random.randint(self.width - 1)]  # anywhere on map
        self.agent2_pos = [np.random.randint(self.height - 1), np.random.randint(self.width - 1)]  # anywhere on map
        # Regenerate positions if overlap
        while self.agent1_pos == self.agent2_pos:
            self.agent1_pos = [np.random.randint(self.height - 1), np.random.randint(self.width - 1)]  # anywhere on map
            self.agent2_pos = [np.random.randint(self.height - 1), np.random.randint(self.width - 1)]  # anywhere on map

        # Moving towards escaper by chaser
        self.toward = False
        # Moving away from chaser by escaper
        self.away = False

        # Accumulated rewards in this episode.
        self.agent1_R = 0.0
        self.agent2_R = 0.0

        # Reset agent1's visited fields.
        self.agent1_visited_fields = set([tuple(self.agent1_pos)])
        self.agent2_visited_fields = set([tuple(self.agent2_pos)])

        # How many timesteps have we done in this episode.
        self.timesteps = 0

        # Return the initial observation in the new episode.
        obs_initial = self._get_discrete_obs()
        self.obs_hist = []
        self.obs_hist.append(obs_initial)
        return self._get_obs()

    def step(self, action: dict):
        """
        Returns (next observation, rewards, dones, infos) after having taken the given actions.

        e.g.
        `action={"agent1": action_for_agent1, "agent2": action_for_agent2}`
        """

        # increase our time steps counter by 1.
        self.timesteps += 1
        # An episode is "done" when we reach the time step limit.
        is_done = self.timesteps >= self.timestep_limit

        # preallocate infos dict for events and positions
        infos = {"agent1": {"events": [], "pos": []}, "agent2": {"events": [], "pos": []}}

        # Agent2 always moves first.
        events = self._move(self.agent2_pos, action["agent2"], self.obs_hist[-1], is_agent1=False)
        events |= self._move(self.agent1_pos, action["agent1"], self.obs_hist[-1], is_agent1=True)

        infos["agent1"]["events"].append(events)
        infos["agent2"]["events"].append(events)

        # Useful for rendering.
        self.collision = "collision" in events
        self.R1_new_field = "agent1_new_field" in events
        self.escape_far = "escape_far" in events
        self.escape_near = "escape_near" in events
        self.escape_close = "escape_close" in events
        self.approach = "approach" in events
        self.R2_new_field = "agent2_new_field" in events

        # Get observations (based on new agent positions).
        obs = self._get_obs()
        # append current observations into the set of observation
        self.obs_hist.append(self._get_discrete_obs())

        # Determine rewards based on the collected events:
        r1 = -1.0 if "collision" in events else 1.0 if "agent1_new_field" in events else -0.5
        r2 = 1.0 if "collision" in events else 0.1 if "agent2_new_field" in events else -0.1

        self.agent1_R += r1
        self.agent2_R += r2
        self.r1 = r1
        self.r2 = r2

        rewards = {
            "agent1": r1,
            "agent2": r2,
        }

        # Generate a `done` dict (per-agent and total).
        dones = {
            "agent1": is_done,
            "agent2": is_done,
            # special `__all__` key indicates that the episode is done for all agents.
            "__all__": is_done,
        }

        return obs, rewards, dones, infos  # <- info dict (not needed here).

    def _get_discrete_obs(self):
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        ag1_discrete_pos = self.agent1_pos[0] * self.width + \
                           (self.agent1_pos[1] % self.width)
        ag2_discrete_pos = self.agent2_pos[0] * self.width + \
                           (self.agent2_pos[1] % self.width)
        return {
            "agent1": np.array([ag1_discrete_pos, ag2_discrete_pos]),
            "agent2": np.array([ag1_discrete_pos, ag2_discrete_pos]),
        }

    def _get_obs(self):
        """
        Returns obs dict (agent name to discrete-pos tuple) using each
        agent's current x/y-positions.
        """
        # create 0 state tensor with the padding ((10+n) x (10+n) x 3)
        padd_n = (self.vision_field - 1)
        vision_side = int(padd_n / 2)

        a1_new_x = self.agent1_pos[0] + vision_side
        a1_new_y = self.agent1_pos[1] + vision_side
        a2_new_x = self.agent2_pos[0] + vision_side
        a2_new_y = self.agent2_pos[1] + vision_side

        state_tensor = np.zeros((self.width + padd_n, self.height + padd_n, 3), dtype=int)
        # Set agent 1 position
        state_tensor[a1_new_x, a1_new_y, 0] = 1
        # Set agent 2 position
        state_tensor[a2_new_x, a2_new_y, 1] = 1

        # Slice state tensor for each agent to get observation
        x1_left = a1_new_x - vision_side
        x1_right = a1_new_x + vision_side + 1
        y1_up = a1_new_y - vision_side
        y1_down = a1_new_y + vision_side + 1

        x2_left = a2_new_x - vision_side
        x2_right = a2_new_x + vision_side + 1
        y2_up = a2_new_y - vision_side
        y2_down = a2_new_y + vision_side + 1

        a1_obs = state_tensor[x1_left:x1_right, y1_up:y1_down, :]
        a2_obs = state_tensor[x2_left:x2_right, y2_up:y2_down, [1, 0, 2]]

        # get discrete positions
        ag1_discrete_pos = self.agent1_pos[0] * self.width + \
                           (self.agent1_pos[1] % self.width)
        ag2_discrete_pos = self.agent2_pos[0] * self.width + \
                           (self.agent2_pos[1] % self.width)

        # provide observation
        # agent_1
        a1_discrete_obs = np.zeros([200, 1], dtype=int)
        a1_discrete_obs[ag1_discrete_pos, 0] = 1

        if (np.any(a1_obs[:, :, 1])):
            a1_discrete_obs[ag2_discrete_pos + 100, 0] = 1

        # agent_2
        a2_discrete_obs = np.zeros([200, 1], dtype=int)
        a2_discrete_obs[ag2_discrete_pos, 0] = 1

        if (np.any(a2_obs[:, :, 1])):
            a2_discrete_obs[ag1_discrete_pos + 100, 0] = 1

        return {
            "agent1": np.array(a1_discrete_obs),
            "agent2": np.array(a2_discrete_obs),
        }

    def _move(self, coords, action, part_coord, is_agent1):
        """
        Moves an agent (agent1 iff is_agent1=True, else agent2) from `coords` (x/y) using the
        given action (0=up, 1=right, etc..) and returns a resulting events dict:
        Agent1: "new" when entering a new field. "bumped" when having been bumped into by agent2.
        Agent2: "bumped" when bumping into agent1 (agent1 then gets -1.0).
        """
        orig_coords = coords[:]
        # if agent 2, sample random actions
        if not is_agent1:
            action = np.random.randint(4)

        # Change the row: 0=up (-1), 2=down (+1)
        coords[0] += -1 if action == 0 else 1 if action == 2 else 0
        # Change the column: 1=right (+1), 3=left (-1)
        coords[1] += 1 if action == 1 else -1 if action == 3 else 0

        # check distance first
        if is_agent1:
            xx = part_coord['agent1'][1]
            if xx < 10:
                prev_coord = [0, xx]
            else:
                prev_coord = [int(i) for i in str(xx)]
            new_dist = np.linalg.norm(np.array(coords) - np.array(prev_coord))
            old_dist = np.linalg.norm(np.array(orig_coords) - np.array(prev_coord))
            if new_dist > old_dist:
                self.away = True
            else:
                self.away = False
        elif not is_agent1:
            xx = part_coord['agent1'][0]
            if xx < 10:
                prev_coord = [0, xx]
            else:
                prev_coord = [int(i) for i in str(xx)]
            new_dist = np.linalg.norm(np.array(coords) - np.array(prev_coord))
            old_dist = np.linalg.norm(np.array(orig_coords) - np.array(prev_coord))
            if new_dist < old_dist:
                self.toward = True
            else:
                self.toward = False

        # Solve collisions.
        # Make sure, we don't end up on the other agent's position.
        # If yes, don't move (we are blocked).
        if (is_agent1 and coords == self.agent2_pos) or (not is_agent1 and coords == self.agent1_pos):
            coords[0], coords[1] = orig_coords
            # Agent2 blocked agent1 (agent1 tried to run into agent2)
            # OR Agent2 bumped into agent1 (agent2 tried to run into agent1)
            return {"collision"}

        # No agent blocking -> check walls.
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.height:
            coords[0] = self.height - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.width:
            coords[1] = self.width - 1

        # If agent1 -> "new" if new tile covered.
        if is_agent1 and not tuple(coords) in self.agent1_visited_fields:
            self.agent1_visited_fields.add(tuple(coords))
            return {"agent1_new_field"}

        # If agent2 -> "new" if new tile covered.
        if not is_agent1 and not tuple(coords) in self.agent2_visited_fields:
            self.agent2_visited_fields.add(tuple(coords))
            return {"agent2_new_field"}

        thr1 = 5
        thr2 = 3
        if is_agent1:
            xx = part_coord['agent1'][1]
            if xx < 10:
                prev_coord = [0, xx]
            else:
                prev_coord = [int(i) for i in str(xx)]
            new_dist = np.linalg.norm(np.array(coords) - np.array(prev_coord))
            old_dist = np.linalg.norm(np.array(orig_coords) - np.array(prev_coord))
            if new_dist >= thr1 and new_dist > old_dist:
                return {"escape_far"}
            elif new_dist >= thr2 and new_dist > old_dist:
                return {"escape_near"}
            elif thr2 > new_dist > old_dist:
                return {"escape_close"}
        elif not is_agent1:
            xx = part_coord['agent1'][0]
            if xx < 10:
                prev_coord = [0, xx]
            else:
                prev_coord = [int(i) for i in str(xx)]
            new_dist = np.linalg.norm(np.array(coords) - np.array(prev_coord))
            old_dist = np.linalg.norm(np.array(orig_coords) - np.array(prev_coord))
            if new_dist < old_dist:
                return {"approach"}

        # No new tile for either agent.
        return set()

    # alternative render with more information
    def render_to_image(self):
        # Define some constants for the image
        FONT_PATH = "arial.ttf"  # Replace with the path to your desired font file
        FONT_SIZE = 14
        FONT_COLOR = (0, 0, 0)  # Black color for text
        CELL_SIZE = 20
        BORDER_SIZE = 1
        BORDER_COLOR = (125, 125, 125)  # Black color for borders

        # Create a new image with the desired dimensions
        image_width = (self.width * CELL_SIZE) + ((self.width + 1) * BORDER_SIZE)
        image_height = (self.height + 13 * CELL_SIZE) + ((self.height + 13 + 1) * BORDER_SIZE)
        image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))

        # Create a drawing context for the image
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        font_text = ImageFont.truetype(FONT_PATH, 12)

        # Add the text labels
        label_x = BORDER_SIZE
        label_y = (self.height * CELL_SIZE) + ((self.height + 1) * BORDER_SIZE) + BORDER_SIZE
        label_text_r1 = "!!Collision!!" if self.collision else "Escape_Near" if self.escape_near else "Escape_Far" if self.escape_far else "Escape" if self.escape_close else "R1_New_Field" if self.R1_new_field else ""
        color_text_r1 = (255, 0, 0) if self.collision else (0, 255, 0) if (
                self.escape_near or self.escape_far or self.escape_close) else (
            255, 255, 255) if self.R1_new_field else (0, 0, 0)
        label_text_r2 = "!!Collision!!" if self.collision else "R2_New_Field" if self.R2_new_field else "Approach" if self.approach else ""
        color_text_r2 = (255, 0, 0) if self.collision else (255, 255, 255) if self.R2_new_field else (
            155, 0, 0) if self.approach else (0, 0, 0)
        draw.text((label_x, label_y), label_text_r1, fill=FONT_COLOR, font=font_text)
        draw.text((label_x + 100, label_y), label_text_r2, fill=FONT_COLOR, font=font_text)

        # add rewards
        if self.r1 < 0:
            c1 = 155
        else:
            c1 = 255

        if self.r2 < 0:
            c2 = 155
        else:
            c2 = 255
        # Agent 1 rewards
        draw.text((label_x, label_y + 15), "E={: .1f}".format(self.agent1_R), fill=FONT_COLOR, font=font_text)
        draw.rectangle((label_x + 115, label_y + 15, label_x + 115 + self.r1 * 20, label_y + 30), fill=(0, 0, c1),
                       outline=BORDER_COLOR)

        draw.text((label_x, label_y + 15 + 15), "C={: .1f}".format(self.agent2_R), fill=FONT_COLOR, font=font_text)
        draw.rectangle((label_x + 115, label_y + 30, label_x + 115 + self.r2 * 20, label_y + 45), fill=(c2, 0, 0),
                       outline=BORDER_COLOR)
        # Agent 2 rewards
        # add timesteps
        draw.text((label_x, label_y + 45), "ts={: .1f}".format(self.timesteps), fill=FONT_COLOR, font=font_text)
        # add arena name
        draw.text((label_x, label_y + 60), self.__class__.__name__, fill=FONT_COLOR, font=font_text)

        for r in range(self.height):
            for c in range(self.width):
                cell_left = (c * CELL_SIZE) + ((c + 1) * BORDER_SIZE)
                cell_top = (r * CELL_SIZE) + ((r + 1) * BORDER_SIZE)
                cell_right = cell_left + CELL_SIZE
                cell_bottom = cell_top + CELL_SIZE

                # Check if tiles have been painted
                if (r, c) in self.agent1_visited_fields and (r, c) in self.agent2_visited_fields:
                    fill_color = (255, 150, 255)  # purple for both painted
                elif (r, c) in self.agent1_visited_fields and (r, c) not in self.agent2_visited_fields:
                    fill_color = (0, 150, 255)  # blue for R1 painted
                elif (r, c) not in self.agent1_visited_fields and (r, c) in self.agent2_visited_fields:
                    fill_color = (255, 150, 100)  # red for R2 painted
                else:
                    fill_color = (255, 255, 255)  # White color for empty cells

                draw.rectangle((cell_left, cell_top, cell_right, cell_bottom), fill=fill_color, outline=BORDER_COLOR)

                # Draw the agent symbols
                hist_pos = self.obs_hist[-5:-1]
                col_prc = [0.4, 0.3, 0.2, 0.1]
                for i in range(0, min(len(hist_pos), 4), 1):
                    tmp = int(255 * col_prc[i])
                    col = (tmp, tmp, tmp)

                    r_a1 = int(hist_pos[i]['agent1'][0] / 10)
                    c_a1 = hist_pos[i]['agent1'][0] % 10

                    r_a2 = int(hist_pos[i]['agent1'][1] / 10)
                    c_a2 = hist_pos[i]['agent2'][1] % 10

                    if [r_a1, c_a1] == [r, c]:  # and self.agent2_pos != [r,c]:
                        draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "x", fill=col,
                                  font=font_text, anchor="mm")
                    if [r_a2, c_a2] == [r, c]:  # and self.agent1_pos != [r,c]:
                        draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "o", fill=col,
                                  font=font_text, anchor="mm")

                if self.agent1_pos == [r, c]:
                    draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "X", fill=color_text_r1,
                              font=font, anchor="mm")
                elif self.agent2_pos == [r, c]:
                    draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "O", fill=color_text_r2,
                              font=font, anchor="mm")

        return np.array(image)

    # prettier render
    def render_to_image2(self):
        FONT_PATH = "Windows\Fonts\seguiemj.ttf"
        FONT_SIZE = 15
        CELL_SIZE = 20
        BORDER_SIZE = 1
        BORDER_COLOR = (125, 125, 125)
        HIGHLIGHT_COLOR = (255, 150, 255)  # Purple for both agents painted
        COLLISION_CAPTION = "!Collision!" if self.collision else "👀"
        CAPTION_COLOR = (255, 0, 0) if self.collision else (0, 0, 0)

        # Lighter shade of the same color
        LIGHTER_COLOR = tuple(min(255, c + 50) for c in HIGHLIGHT_COLOR)

        image_width = (self.width * CELL_SIZE) + ((self.width + 1) * BORDER_SIZE)
        image_height = (self.height * CELL_SIZE) + (
                (self.height + 1) * BORDER_SIZE) + FONT_SIZE + 5  # Additional height for caption
        image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        font_text = ImageFont.truetype(FONT_PATH, 12)

        # Draw collision caption
        caption_bbox = draw.textbbox((0, 0), COLLISION_CAPTION, font=font_text)
        caption_x = (image_width - caption_bbox[2]) // 2
        caption_y = image_height - FONT_SIZE
        draw.text((caption_x, caption_y), COLLISION_CAPTION, fill=CAPTION_COLOR, font=font_text)

        for r in range(self.height):
            for c in range(self.width):
                cell_left = (c * CELL_SIZE) + ((c + 1) * BORDER_SIZE)
                cell_top = (r * CELL_SIZE) + ((r + 1) * BORDER_SIZE)
                cell_right = cell_left + CELL_SIZE
                cell_bottom = cell_top + CELL_SIZE

                if (r, c) in self.agent1_visited_fields and (r, c) in self.agent2_visited_fields:
                    fill_color = LIGHTER_COLOR   # Both agents painted
                elif (r, c) in self.agent1_visited_fields:
                    fill_color = (50, 200, 255)  # Blue for agent 1 painted
                elif (r, c) in self.agent2_visited_fields:
                    fill_color = (255, 200, 150)  # Red for agent 2 painted
                else:
                    fill_color = (255, 255, 255)

                # Draw cell with rounded corners
                draw.rounded_rectangle([(cell_left, cell_top), (cell_right, cell_bottom)], 5, fill=fill_color,
                                       outline=BORDER_COLOR)

                # Draw agent symbols with color based on collision
                emoji_color = (255, 0, 0) if self.collision and (
                        [r, c] == self.agent1_pos or [r, c] == self.agent2_pos) else (0, 0, 0)
                if [r, c] == self.agent1_pos:
                    draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "🐭", fill=emoji_color, font=font,
                              anchor="mm")
                elif [r, c] == self.agent2_pos:
                    draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "🐱", fill=emoji_color, font=font,
                              anchor="mm")

        return np.array(image)
