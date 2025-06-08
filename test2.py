import pytesseract
from PIL import Image

image = Image.open("seed_label.jpg")
text = pytesseract.image_to_string(image, lang='eng')
print(text)
