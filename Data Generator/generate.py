# pip install Pillow
# pip install numpy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint, choice, uniform
import string
import time

colors = {
  "black": 0x1c1c1c,
  "white": 0xfcfcfc,
}
image_width = 250
image_height = 70
font_size = 50
characters = [5,5]

def get_text():
  out_string = ""
  for i in range(randint(characters[0], characters[1])):
    out_string += choice(string.ascii_letters+"0123456789")
  return out_string

def draw_pixel(draw,x,y, thickness):
  if(thickness > 1):
    draw.line([(x, y), (x+thickness*([1,-1][randint(0, 1)]),y+thickness*([1,-1][randint(0, 1)]))], fill=colors["black"], width=thickness)
  else:
    draw.line([(x, y), (x,y)], fill=colors["black"])

def add_noise(draw, amount, thickness):
  for i in range(int(amount)):
    draw_pixel(draw, randint(0, image_width), randint(0, image_height), thickness)

def add_lines(draw, amount, thickness):

  thickness = 1
  wiggle_room_thickness = 2

  for i in range(int(amount)):
    wiggle_thickness = randint(thickness,thickness+wiggle_room_thickness)
    draw.line([(randint(0, image_width), randint(0, image_height)), (randint(0, image_width),randint(0, image_height))], fill=colors["black"],width=wiggle_thickness,)

def draw_characters(draw, image, text):
  
  fonts = ["ComicSansMS3.ttf","carbontype.ttf","Kingthings_Trypewriter_2.ttf","Sears_Tower.ttf","TravelingTypewriter.ttf"]
  wiggle_room_width = 48 - len(text) * 6
  wiggle_room_height = 13
  width_padding = 30
  font_size = 30
  wiggle_room_font_size = 10
  rotation_degrees = 20
  wiggle_room_rotation_percent = 0.4

  spacing_width = (image_width-width_padding) / len(text)
  next_letter_pos = width_padding/2


  for character in text:
    wiggle_width = uniform(0,wiggle_room_width)
    wiggle_height = uniform(0,wiggle_room_height)
    wiggle_font_size = uniform(font_size - wiggle_room_font_size / 2, font_size  + wiggle_room_font_size / 2)
    wiggle_rotation_degrees = rotation_degrees * uniform( -wiggle_room_rotation_percent,  wiggle_room_rotation_percent)
    font = ImageFont.truetype("./fonts/"+fonts[randint(0,len(fonts)-1)], int(wiggle_font_size))
    character_image = Image.new("RGB", (image_height, image_height), 255)
    draw = ImageDraw.Draw(character_image)
    draw.rectangle([0, 0, image_height, image_height], fill=colors["white"])
    draw.text((wiggle_width, wiggle_height), character, fill=colors["black"], font=font)
    character_image = character_image.rotate(wiggle_rotation_degrees, expand=True, fillcolor="white")
    image.paste(character_image, (int(next_letter_pos), 0))
    next_letter_pos += spacing_width
  

num_samples = input("How many images do you want to generate?")
text_done = []
start = time.time()

for i in range(int(num_samples)):

  image = Image.new('RGB',(image_width, image_height))
  draw = ImageDraw.Draw(image)
  draw.rectangle([0, 0, image_width, image_height], fill=colors["white"])

  text = get_text()
  if(text not in text_done):
    text_done.append(text)
  else:
    print("Skipped")
    continue
  draw_characters(draw, image, text)

  add_noise(draw, amount=int(image_width*image_height*0.001), thickness = 1)
  add_noise(draw, amount=int(image_width*image_height*0.001), thickness = 2)
  add_noise(draw, amount=int(image_width*image_height*0.001), thickness = 3)
  add_lines(draw, amount=1, thickness=2)
  add_lines(draw, amount=1, thickness=3)


  image.save("./output/"+text+".jpg")
end = time.time()
print("Operation completed. Time taken: "+str(end - start)+" seconds")