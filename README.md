# Telegram_bot_with_style_transfering_model

This project implements a python telegram bot for image style transfering. 
You can send to the bot 2 images and get as a result one image (the first one) with some style (from the second one).

## Example
<b> Original image

<p align="center">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/original.jpg" height="250" title="Original photo" alt="Original photo">
</p>

<b> Style 1 and result

<p align="center">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_1.png" height="250" title="Style 1" alt="Style 1">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_1_result.png" height="250" title="Style 1 RESULT" alt="Style 1 RESULT">
</p>

<b> Style 2 and result

<p align="center">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_2.png" height="250" title="Style 2" alt="Style 2">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_2_result.png" height="250" title="Style 2 RESULT" alt="Style 2 RESULT">
</p>

<b> Style 3 and result

<p align="center">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_3.jpg" height="250" title="Style 3" alt="Style 3">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_3_result.png" height="250" title="Style 3 RESULT" alt="Style 3 RESULT">
</p>

<b> Style 4 and result

<p align="center">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_4.jpeg" height="250" title="Style 4" alt="Style 4">
  <img src="https://github.com/Anastasiyofworld/Telegram_bot_with_style_transfering_model/blob/main/imges/style_4_result.png" height="250" title="Style 4 RESULT" alt="Style 4 RESULT">
</p>


## Instalation

0. Use `Python 3.10`
1. Install the nessessary libraries:
- `pip install python-telegram-bot`
- `pip install scipy`
- `pip install numpy`
- `pip install torch`
- `pip install torch_snippets`
- `pip install Pillow`
- `pip install torchvision`

2. Clone repository
3. Create **your own token** in **BotFather** and add your bot token to this file `token.txt`
4. Run `bot.py` 

## Using

1. After bot started call from your telegram `/start` command
2. Send 2 images
3. Wait, it will be received in around 20 seconds

## Settings

`bot.py`
- For adding more iterations to the model you can change the row `a = stm.Neural_style_transfer(image_size=256, max_iters=50)` (usual enough `max_iters=500`) Just for a test of how bot working use around 1-20
- For changing max result image size change `self.image_size = 128` - it's a  maximum image pixel size by one side
  
