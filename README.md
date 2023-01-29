# Telegram_bot_with_style_transfering_model

instalation

1) pip install ...
2) ..
3) clone repository
4) create token.txt and add your bot token to this file
5) run bot.py 

Using

1) After bot started call from your telegram /start command
2) Send 2 images
3) Wait it will resive around 20 secoind

Settings

style_transfer_model.py
  -for add iterations in model change row "max_iters = 1" (usual enough 250)
    just for test working bot use around 1-20
  -for change max result image size change "self.image_size = 128" its maximum image pixel size by one side
  
