import string
from html2image import Html2Image
from pathlib import Path
import uuid
import cv2
import numpy as np
import os

html_1 = """
<head>
</head>
<body>
<div class="req"> 
<p> 
${expr}
</p>
</div>
</body>
"""
css = """
body {
  margin: 0px;
  padding: 0px;
  background-color: #FFFFFF;
  
}
.req {
    margin: 0px;
    background-color: #5FBB8E;
    color: #1D110B;
    padding: 0px;
    width: ${size_horizontal}px !important;
    font-family: "Lato", sans-serif;
  font-size: 25px;
    text-align: center;
  border: 2px solid #333
}"""

height_max = 700

def get_html_string_image(request_str, size_horizontal, tmp_folder="./tmp/"):
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    hti = Html2Image(size=(size_horizontal * 2, height_max), output_path=tmp_folder)
    html_str = string.Template(html_1).substitute(expr=request_str)
    css_str = string.Template(css).substitute(size_horizontal=size_horizontal)
    # generate random name
    file_name = f"{str(uuid.uuid4())}.png"
    hti.screenshot(html_str=html_str, css_str=css_str, save_as=file_name)
    file_name = os.path.join(tmp_folder, file_name)
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # then crop based on the background color
    for height_index in range(50, img.shape[0]):
        cur_row = img[height_index, :]
        if np.all(cur_row == [255, 255, 255]):
            break
    for width_index in range(50, img.shape[1]):
        cur_col = img[:, width_index]
        if np.all(cur_col == [255, 255, 255]):
            break

    img = img[:height_index, :width_index]

    scale_ratio = size_horizontal / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)

    return img


def rounded_rectangle(src, top_left, bottom_right, radius=1, color=255, thickness=1, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[1], top_left[1])
    p3 = (bottom_right[1], bottom_right[0])
    p4 = (top_left[0], bottom_right[0])

    height = abs(bottom_right[0] - top_left[1])

    if radius > 1:
        radius = 1

    corner_radius = int(radius * (height/2))

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect], 
        [top_left_rect_left, bottom_right_rect_left], 
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src

html_2 = """
<!DOCTYPE html>
<html>
<head>
</head>
<body>
<div class="slider_holder">
    <div class="slidecontainer">
        <input type="range" min="1" max="100" value="${amount}" class="slider" id="myRange">
    </div>
</div>
</div>
</body>
</html>
"""

css_2 = """
body {
  margin: 0px;
  padding: 0px;
  background-color: #FFFFFF;
}
.slider_holder {
    margin: 0px;
    padding-top: 50px;
    padding-bottom: 50px;
    padding-left: 0px;
    padding-right: 0px;
    background-color: #5FBB8E;
    color: #1D110B;
    width: ${size_horizontal}px !important;
    font-family: "Lato", sans-serif;
  font-size: 30px;
    text-align: center;
  border: 2px solid #333
}
.slider {
  -webkit-appearance: none;
  width: 95%;
  height: 20px;
  border-radius: 5px;  
  background: #1D110B;
  outline: none;
  opacity: 1.0;
  -webkit-transition: .2s;
  transition: opacity .2s;
}

.slider::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  width: 55px;
  height: 55px;
  border-radius: 35%; 
  background: #FFF8F0;
  color: #FFF8F0;
  border: 5px solid #E0E2DB;
  cursor: pointer;
}

.slider::-moz-range-thumb {
  width: 25px;
  height: 25px;
  border-radius: 50%;
  background: #E0E2DB;
  cursor: pointer;
}
 """

def get_slider_html_image(cur_value, max_value, size_horizontal, tmp_folder="./tmp/"):
    Path(tmp_folder).mkdir(parents=True, exist_ok=True)
    hti = Html2Image(size=(size_horizontal * 2, height_max), output_path=tmp_folder)
    amount = int(cur_value/max_value * 40) + 50
    html_str = string.Template(html_2).substitute(amount=amount)
    css_str = string.Template(css_2).substitute(size_horizontal=size_horizontal)

    file_name = f"{str(uuid.uuid4())}.png"
    hti.screenshot(html_str=html_str, css_str=css_str, save_as=file_name)
    file_name = os.path.join(tmp_folder, file_name)
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # then crop based on the background color
    for height_index in range(50, img.shape[0]):
        cur_row = img[height_index, :]
        if np.all(cur_row == [255, 255, 255]):
            break
    for width_index in range(50, img.shape[1]):
        cur_col = img[:, width_index]
        if np.all(cur_col == [255, 255, 255]):
            break
    img = img[:height_index, :width_index]
    # scale up image in both dim so that wider is the same as the input
    scale_ratio = size_horizontal / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale_ratio, fy=scale_ratio)
    return img