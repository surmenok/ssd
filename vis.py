"""
Based on code from yad2k project: https://github.com/allanzelener/YAD2K
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import colorsys
import IPython.display


class Vis:
    def __init__(self, class_names):
        self.class_names = class_names
        self.colors = self._generate_colors(class_names)

    def _generate_colors(self, class_names):
        hsv_tuples = [(x / len(class_names), 1., 1.)
                      for x in range(len(class_names))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))

        prev_state = np.random.get_state()
        np.random.seed(42)
        np.random.shuffle(colors)
        np.random.set_state(prev_state)

        return {class_names[i]: v for i, v in enumerate(colors)}

    def draw_rectangle(self, draw, x0, y0, x1, y1, thickness, color):
        for i in range(thickness):
            draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=color)

    def show_image(self, image, format='png'):
        f = BytesIO()
        img = Image.fromarray(image)
        img.save(f, format)
        IPython.display.display(IPython.display.Image(data=f.getvalue()))

    def show_image_with_boxes(self, image, boxes, labels, format='png',
                              font_path='../resources/font/FiraMono-Medium.otf'):
        f = BytesIO()
        img = Image.fromarray(image)
        draw = ImageDraw.Draw(img)

        font = ImageFont.truetype(
            font=font_path,
            size=20)

        for i in range(labels.shape[0]):
            label = self.class_names[labels[i]] if labels[i] < len(self.class_names) else 'Bg'
            color = self.colors[label] if label in self.colors else (255, 255, 255)
            top, left, bottom, right = boxes[i, :]
            self.draw_rectangle(draw, left, top, right, bottom, 5, color)

            label_size = draw.textsize(label, font)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=color)

            draw.text(text_origin, label, fill=(0, 0, 0), font=font)

        del draw

        img.save(f, format)
        IPython.display.display(IPython.display.Image(data=f.getvalue()))
