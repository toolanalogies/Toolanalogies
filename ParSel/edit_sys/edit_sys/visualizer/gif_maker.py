

import numpy as np
import time
import os
import threading
import cv2
import open3d as o3d
import open3d.visualization.gui as gui
import sys
from pathlib import Path

from .base import EditSysApp
from .gif_utils import rounded_rectangle, get_html_string_image, get_slider_html_image
from .constants import GIF_W, GIF_H, BLK_SIZE
from edit_sys.shape_system.shape_atoms import Part
from edit_sys.shape_system.edits import MAIN_VAR
N_CYCLES = 5
TIMER_VALUE = 0.75
RANGE = (0, 0.5)
MAX_FRAME_COUNT = 270

HTML_W = GIF_W
HTML_H = BLK_SIZE

class GIFCreator(EditSysApp):

    def __init__(self,
                 edit_request,
                 dataset_index,
                 shape_id,
                 selected_obj,
                 method_marker,
                 data_dir,
                 output_dir,
                 redo_search=False,
                 width=GIF_W,
                 height=GIF_H,):
        super().__init__(dataset_index,
                         shape_id,
                         selected_obj,
                         method_marker,
                         data_dir,
                         output_dir,
                         redo_search,
                         width=width,
                         height=height)
        
        start_gen = True
        self.shape = self.symbolic_data[0]
        self._scene.scene.set_lighting(self._scene.scene.LightingProfile.DARK_SHADOWS, (-0.577, -0.577, -0.077))
        if start_gen:
            self.get_edit(link_to_single=True)
        self.edit_request = edit_request

        self._obj_panel.get_children()[0].set_is_open(False)
        self._chat_panel.get_children()[0].set_is_open(False)
        self._procedural_panel.get_children()[0].set_is_open(False)
        # self._settings_panel.get_children()[0].set_is_open(False)
        
        self.initialize_gifer()

        if start_gen:
            new_mesh = o3d.geometry.TriangleMesh()
            for name, geom in self.name_to_geom.items():
                item_type = name.split("_")[-1]
                if item_type in ['mesh']:
                    for mesh_info in geom.meshes:
                        new_mesh += mesh_info.mesh
            # Set the max range and export that too
            cur_value = self.range[-1]
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = cur_value
                self._edit_execute(symbol)

            for name, geom in self.name_to_geom.items():
                item_type = name.split("_")[-1]
                if item_type in ['mesh']:
                    for mesh_info in geom.meshes:
                        new_mesh += mesh_info.mesh

            cur_value = self.range[0]
            for symbol, caller in self.symbol_value_dict.items():
                self.symbol_value_dict[symbol] = cur_value
                self._edit_execute(symbol)

            # get the center
            self.scene_aabb = new_mesh.get_axis_aligned_bounding_box()
            self.mesh_center = self.scene_aabb.get_center()
            # get AABB
            delta = self.scene_aabb.get_half_extent() * 2
            self.camera_y = self.scene_aabb.get_half_extent()[1] * 1.5
            self.radius = (delta[0] ** 2 + delta[2] ** 2)**0.5
            self.radius = max(self.radius, self.camera_y)
            theta = 0
            camera_x = self.radius * np.sin(theta)
            camera_z = self.radius * np.cos(theta)
            camera_location = self.scene_aabb.get_center() + np.array([camera_x, self.camera_y, camera_z])
            self._scene.scene.camera.look_at(self.mesh_center, camera_location, [0, 1, 0])


            gif_save_dir = os.path.join(self.output_dir, "gifs", self.method_marker, f"{self.dataset_index}")
            Path(gif_save_dir).mkdir(parents=True, exist_ok=True)
            self.gif_save_dir = gif_save_dir
            threading.Thread(target=self.online_gif).start()

    def initialize_gifer(self):
        self.tick_time = time.time()
        self.old_counter = 0
        self.new_counter = 0
        self.max_count = MAX_FRAME_COUNT
        # self.range = RANGE
        self.is_done = False
        self.invert_range = False
        # Update range based on the edit.

        # self.set_range()
            
        self.sym_values, self.theta_values = self.get_values_to_use()
        self.theta_values += np.pi/4
        self.make_post = True
        # gem center
        self.add_prompt = True

    def get_values_to_use(self):
        values = np.arange(0, self.max_count)/self.max_count
        values = (values * N_CYCLES * 2 * np.pi) - np.pi
        b = 2.0
        values = np.sqrt((1 + b**2)/(1 + b**2 * np.cos(values)**2)) * np.cos(values)
        values = (1 + values)/2 * (self.range[1] - self.range[0])+ self.range[0]
        # \ a^{\left(p\right)}\ \cdot\ \operatorname{sign}\left(a\right)\ +\ 1
        # p = 0.5
        # a = np.sin(values)
        if self.invert_range:
            values = - values
        # values = (np.abs(a)**p * np.sign(a) + 1) / 2 * self.range[1] + self.range[0]
        count_per_cycle = self.max_count / N_CYCLES
        n_rots = count_per_cycle * (N_CYCLES -1)
        theta_values = np.arange(0, n_rots)/n_rots
        zeros = np.zeros(int(count_per_cycle)).astype(np.float64)
        theta_values = np.concatenate([zeros, theta_values])
        theta_values = theta_values * 2 * np.pi
        return values, theta_values
    
    def online_gif(self):
        iteration = 0
        while iteration < self.max_count:
            def update():
                if self.old_counter < self.max_count:
                    print(f"updating {self.old_counter}")
                    cur_value = self.sym_values[self.old_counter]
                    for symbol, caller in self.edit_callbacks.items():
                        caller(cur_value)
                    self.old_counter += 1
            def update_and_render():
                if self.old_counter < self.max_count:
                    cur_value = self.sym_values[self.old_counter]
                    for symbol, caller in self.symbol_value_dict.items():
                        self.symbol_value_dict[symbol] = cur_value
                        self._edit_execute(symbol)
                    theta = self.theta_values[self.old_counter]
                    camera_x = self.radius * np.sin(theta)
                    camera_z = self.radius * np.cos(theta)
                    camera_location = np.array([camera_x, self.camera_y, camera_z])
                    self._scene.scene.camera.look_at(self.mesh_center, camera_location, [0, 1, 0])
                    self.window.post_redraw()
                    self._scene.force_redraw()
                    file_name = os.path.join(self.gif_save_dir, f"{self.old_counter:04d}.png")
                    self.export_image(file_name, self._scene.frame.width, self._scene.frame.height)
                    self.old_counter += 1
                    self.make_post = True
            if self.make_post:
                print(f"adding post with {iteration}")
                o3d.visualization.gui.Application.instance.post_to_main_thread(self.window, update_and_render)
                self.make_post = False
                iteration += 1

        print("Done!")
        # Now convert to gif
        # Now for each image add the edit request text to the image using cv2
        # Also add a slider animation with the value

        self.html_stuff()

        # then use ffmpeg to convert the images to a gif using os
        # Path(self.gif_save_dir).mkdir(parents=True, exist_ok=True)
        out_name = os.path.join(self.gif_save_dir, f"{self.dataset_index}.gif")
        # os.system(f"ffmpeg -i {self.gif_save_dir}/%04d.png -vf scale=512:-1 {out_name} -y")
        # os.system(f"ffmpeg -framerate 60 -i {self.gif_save_dir}/%04d.png -vf scale=512:-1 -c:v libx264 -crf 25 -pix_fmt yuv420p {out_name} -y")
        # os.system(f"ffmpeg -framerate 30 -i {self.gif_save_dir}/%04d.png -vf 'fps=15,scale=256:-1:flags=lanczos,palettegen=max_colors=128' -y {self.gif_save_dir}/palette.png")
        os.system(f"ffmpeg -framerate 30 -i {self.gif_save_dir}/%04d.png -vf 'fps=15,scale=512:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse' {out_name} -y")
        # os.system("source ~/.cargo/env")
        # os.system(f"gifski {self.gif_save_dir}/%04d.png -o {out_name}")
        # Remove the images
        os.system(f"rm {self.gif_save_dir}/*.png")
        # clean up ./tmp as well
        os.system(f"rm ./tmp/*")
        # quit the application
        self.is_done = True
        self.window.close()
        gui.Application.instance.quit()

    def html_stuff(self):
        request_str = self.edit_request
        size_horizontal = HTML_W
        str_image = get_html_string_image(request_str, HTML_W)
        str_image = cv2.cvtColor(str_image, cv2.COLOR_BGR2RGB)

        for i in range(self.max_count):
            # in the bottom center
            image_name = os.path.join(self.gif_save_dir, f"{i:04d}.png")
            image = cv2.imread(image_name)
            slider_image = get_slider_html_image(self.sym_values[i], self.range[1], HTML_W)
            slider_image = cv2.cvtColor(slider_image, cv2.COLOR_BGR2RGB)

            # Concatenate the images vertically
            if self.add_prompt:
                final_image = np.concatenate((image, slider_image, str_image), axis=0)
            else:
                final_image = np.concatenate((image, slider_image), axis=0)
            cv2.imwrite(image_name, final_image)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image
            quality = 9  # png
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)
    
    def online_scrolling_viewer(self):

        while not self.is_done:
            new_time = time.time() - self.tick_time
            def update_amount():
                z = np.random.uniform(-0.5, 0.5)
                for symbol, caller in self.edit_callbacks.items():
                    caller(z)

            o3d.visualization.gui.Application.instance.post_to_main_thread(self.window, update_amount)

    def get_edit_text_bit(self, ):


        text_img = np.ones((HTML_H, HTML_W, 3), dtype=np.uint8) * self.text_bg
        # get boundary of this text
        textsize = cv2.getTextSize(self.edit_request, self.font, self.font_size, self.font_color, self.font_thickness)[0]

        # get coords based on boundary
        textX = (text_img.shape[1] - textsize[0]) / 2
        textY = (text_img.shape[0] + textsize[1]) / 2

        # add text centered on image
        cv2.putText(text_img, self.edit_request, (textX, textY), self.font, self.font_size, self.font_color, self.font_thickness)
        return text_img

    def get_slider_image(self, amount, max_limit):
        slider_bg = np.array([95, 187, 151]).astype(np.uint8)
        slider_color = (29, 17, 11)
        circle_color = (255, 248, 240)
        circle_color_2 = (224, 226, 219)
        border_color = (55, 55, 55)
        slider_radius = 0.5
        thickness = -1
        circle_radius = 25
        circle_radius_2 = 30

        slider_img = np.ones((HTML_H, HTML_W, 3), dtype=np.uint8) * slider_bg
        slider_img = slider_img.copy()

        # image_size = (HTML_H, HTML_W, 3)
        # slider_img = np.ones(image_size)
        slider_size = (HTML_H * 0.2, HTML_W * 0.9)
        top_left = (HTML_H - slider_size[0], HTML_W - slider_size[1])

        top_left = (int(top_left[0]/2), int(top_left[1]/2))
        bottom_right = (top_left[0] + slider_size[0], top_left[1] + slider_size[1])
        bottom_right = (int(bottom_right[0]), int(bottom_right[1]))

        slider_img_with_bar = rounded_rectangle(slider_img, top_left[::-1], bottom_right, 
                                                radius=slider_radius, color=slider_color, thickness=thickness)
        center_coordinate = (HTML_W//2, HTML_H//2)
        delta = (amount/max_limit) * 0.9 * (HTML_H//2)
        center_coordinate = (center_coordinate[0] , center_coordinate[1] + int(delta)) 

        slider_img_with_circle = cv2.circle(slider_img_with_bar, center_coordinate, 
                                            radius=circle_radius_2, color=circle_color_2, thickness=thickness)
        slider_img_with_circle = cv2.circle(slider_img_with_circle, center_coordinate, 
                                            radius=circle_radius, color=circle_color, thickness=thickness)
        return slider_img_with_circle
