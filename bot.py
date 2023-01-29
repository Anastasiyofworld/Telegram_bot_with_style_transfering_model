import logging
import telegram
import asyncio
import concurrent.futures
import functools
import os
import style_transfer_model as stm
from multiprocessing import Manager
from telegram import Update
from telegram.ext import Application, ContextTypes, CallbackContext, \
    CommandHandler, filters, \
    MessageHandler

import PIL


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO

)
logger = logging.getLogger(__name__)

num_of_pics_for_user = Manager().dict()


async def help_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text('''
    Бот может добавить стиль к вашему изображению.
    Пожалуйста, отправьте два изображения в следующем порядке:
    1 - изображение, к которому будет применен стиль
    2 - изображение стиля

    Допустим, вы хотите получить свою фотографию в стиле живописи Ван Гога.
    Отправьте боту первым избражением свою фотографию, а вторым - любимый шедевр.

    ''')


async def start_command(update: Update, context: CallbackContext) -> None:
    num_of_pics_for_user[update.effective_chat.id] = 0
    await update.message.reply_text('''
    Бот может добавить стиль к вашему изображению.
    Пожалуйста, отправьте два изображения в следующем порядке:
    1 - изображение, к которому будет применен стиль
    2 - изображение стиля
    ''')



async def receive_image(update: Update, context: CallbackContext):
    user = update.message.from_user
    photo_file = await update.message.photo[-1].get_file()
    await photo_file.download_to_drive(f"user_photo_"
                                       f"{update.effective_chat.id}_"
                                       f"{num_of_pics_for_user[update.effective_chat.id]}.jpg")
    num_of_pics_for_user[update.effective_chat.id] += 1
    if num_of_pics_for_user[update.effective_chat.id] == 2:
        await update.message.reply_text('Начинаю создавать новое '
                                        'изображение. Это может занять '
                                        'несколько минут.')
        asyncio.create_task(process_image(update,update.effective_chat.id))
        logger.info("Thread moved ")
        return
    logger.info("Photo of %s: %s", user.first_name, "user_photo.jpg")
    await update.message.reply_text("Спасибо за изображения!")

def sendAnswerImage( chat_id, result):
    logger.info("Start sending answer")
    result.save(f"user_photo"
                        f"_{chat_id}_result.jpg")
    num_of_pics_for_user[chat_id] = -1
    logger.info("End async function")


async def process_image(update: Update, chat_id):
    logger.info("Start async function")
    try:
         os.remove(f"user_photo_{chat_id}_result.jpg")
    except FileNotFoundError:
        print("Can't delete prev image.")
    a = stm.Neural_style_transfer(image_size=256, max_iters=300)

    a.set_images(f"user_photo_{chat_id}_0.jpg",
                 f"user_photo_{chat_id}_1.jpg")
    callback_with_args = functools.partial(sendAnswerImage, chat_id=chat_id)
    a.start_modeling(callback=callback_with_args)
    tries = 600
    while num_of_pics_for_user[chat_id] != -1 :
        await asyncio.sleep(1)
        if tries <= 0:
            await update.message.reply_text('Превышено автоматическое ожидание ответа. Попробуйте спросить результат через какое-то время командой /result')
            return
        tries-=1
    num_of_pics_for_user[chat_id] = 0
    await update.message.reply_photo(f"user_photo_{chat_id}_result.jpg")
    await update.message.reply_text('Наслаждайтесь результатом!')

async def result_image(update: Update, context: CallbackContext):

    file = f"user_photo_{update.effective_chat.id}_result.jpg"
    if os.path.exists(file):
        await update.message.reply_photo(file)
        await update.message.reply_text('Наслаждайтесь результатом!')
    else:
        await update.message.reply_text('Изображения еще в обработке... подождите пожалуйста.')

def main() -> None:
    application = Application.builder().token(TOKEN).concurrent_updates(True).build()

    application.add_handler(CommandHandler('help', help_command, block=False))
    application.add_handler(CommandHandler('start', start_command, block=False))
    application.add_handler(CommandHandler('result', result_image, block=False))
    image_handler = MessageHandler(filters.PHOTO, receive_image, block=False)
    application.add_handler(image_handler)
    print('Started')
    application.run_polling()



if __name__ == '__main__':

    with open('token.txt') as f:
        TOKEN = f.read()

    main()

