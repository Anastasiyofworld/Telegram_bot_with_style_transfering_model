# Telegram_bot_with_style_transfering_model

This project implements a python telegram bot for image style transfering. 
You can send to the bot 2 images and get as a result one image (the first one) with some style (from the second one).

## Example
**[Original photo](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/original.jpg)**


**[Style 1](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_1.png)**
**[Style 1 RESULT](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_1_result.png)**


**[Style 2](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_2.png)**
**[Style 2 RESULT](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_2_result.png)**


**[Style 3](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_3.jpg)**
**[Style 3 RESULT](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_3_result.png)**



**[Style 4](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_4.jpeg)**
**[Style 4 RESULT](https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_4_result.png)**


## Instalation

0. Use Python 3.10
1. 
- pip install python-telegram-bot
- pip install scipy
- pip install numpy
- pip install torch
- pip install torch_snippets
- pip install Pillow
- pip install torchvision

2. clone repository
3. create **your own token** in BotFather and add your bot token to this file `token.txt`
4. run `bot.py` 

## Using

1. After bot started call from your telegram /start command
2. Send 2 images
3. Wait, it will be resived in around 20 seconds

## Settings

`bot.py`
 - for add more iterations to the model you can change the row `a = stm.Neural_style_transfer(image_size=256, max_iters=50)` (usual enough `max_iters=500`) Just for a test of how bot working use around 1-20
- for change max result image size change `self.image_size = 128` - tt's a  maximum image pixel size by one side
  
