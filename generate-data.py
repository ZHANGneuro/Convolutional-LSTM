import pandas as pd
# from PIL import Image
# import os



from PIL import Image
import os
# make sure the "star.png" image exists
assert os.path.exists('star.jpg'), "star.jpg not found"
# create the base image and the star image
base = Image.new('RGB', (90, 40), color='white')
star = Image.open('star.jpg').resize((4, 4))

# dictionary to hold the positions
ima_height = 18
positions = {1: (15-2, ima_height), 2: (45-2, ima_height), 3: (75-2, ima_height)}

# create the 30 images
img = base.copy()
img.paste(star, positions[3])
img.save('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/images_identity/3.png')





from PIL import Image
import os
# make sure the "star.png" image exists
assert os.path.exists('star.jpg'), "star.jpg not found"
# create the base image and the star image
base = Image.new('RGB', (90, 40), color='white')
star = Image.open('star.jpg').resize((18, 18))

# dictionary to hold the positions
ima_height = 11
positions = {1: (15-9, ima_height), 2: (45-9, ima_height), 3: (75-9, ima_height)}

# create the 30 images
img = base.copy()
img.paste(star, positions[1])
img.save('/Users/bo/Documents/PycharmProjects/dingwei_relational_memory/images_identity/1.png')


