"""Headless schematic GIFs from Newton link poses, without a graphics context."""

from itertools import product
from math import ceil, isqrt

from PIL import Image, ImageDraw

from isaaclab.utils.math import quat_apply


class GraspGif:
    """Draw an oblique view of each tabletop and the simulated link positions."""

    width = 420
    height = 360

    def __init__(self, env):
        self.env = env
        self.frames = []
        self.cols = isqrt(env.num_envs - 1) + 1
        self.rows = ceil(env.num_envs / self.cols)
        names = [f"panda_link{i}" for i in range(1, 8)]
        names += ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
        self.links = env.robot.find_bodies(names, preserve_order=True)[0]
        self.corners = env.actions.new_tensor(list(product((-0.025, 0.025), repeat=3)))

    def capture(self, tick):
        position = self.env.robot.data.body_link_pose_w.torch[:, self.links, :3]
        origins = self.env.scene.env_origins[:, None, :]
        links = (position - origins).tolist()
        pose = self.env.scene["object"].data.root_link_pose_w.torch
        corners = quat_apply(pose[:, None, 3:7].expand(-1, 8, -1), self.corners.expand(len(links), -1, -1))
        cubes = (corners + pose[:, None, :3] - origins).tolist()
        frame = Image.new("RGB", (self.cols * self.width, self.rows * self.height), "#111b2a")
        draw = ImageDraw.Draw(frame)
        for index, (arm, cube) in enumerate(zip(links, cubes, strict=True)):
            x0, y0 = index % self.cols * self.width, index // self.cols * self.height

            def project(point):
                x, y, z = point
                return (x0 + 225 + 145 * x + 110 * y, y0 + 300 + 90 * x - 80 * y - 225 * z)

            draw.rectangle((x0 + 4, y0 + 4, x0 + self.width - 5, y0 + self.height - 5), fill="#172638")
            table = [project((x, y, 0.255)) for x, y in [(-0.9, -0.75), (-0.1, -0.75), (-0.1, 0.75), (-0.9, 0.75)]]
            draw.polygon(table, fill="#334a5d", outline="#7091a5", width=2)
            for v, vertex in enumerate(cube):
                for axis in (1, 2, 4):
                    if not v & axis:
                        draw.line((project(vertex), project(cube[v | axis])), fill="#ffb172", width=3)
            chain = [project((0, 0, 0))] + [project(point) for point in arm[:8]]
            draw.line(chain, fill="#d7e7f5", width=9, joint="curve")
            for x, y in chain:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#74c8dc")
            for finger in arm[8:]:
                draw.line((chain[-1], project(finger)), fill="#74c8dc", width=4)
            draw.text((x0 + 16, y0 + 16), f"FRANKA PANDA  |  ENV {index + 1:02d}", fill="#eff7ff")
            draw.text((x0 + 16, y0 + 34), f"Schematic pose replay  |  t = {tick / 120:.2f} s", fill="#a8c7d9")
        self.frames.append(frame)

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(path, save_all=True, append_images=self.frames[1:], duration=50, loop=0)
        print(f"Saved {len(self.frames)} GIF frames to {path}", flush=True)
