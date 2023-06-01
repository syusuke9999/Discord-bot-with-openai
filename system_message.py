import datetime
import pytz
from enum import Enum


class Topic(Enum):
    General_Discord_Bot = 1
    Online_Game = 2


class SystemMessage:
    def __init__(self, topic=None):
        self.topics = topic
        if topic is None:
            self.topics = 'None'
            self.system_message_content = 'You are useful assistant.'
        else:
            self.system_message_content = self.set_system_message_content()

    def set_system_message_content(self):
        if self.topics == Topic.General_Discord_Bot:
            jst = pytz.timezone('Asia/Tokyo')
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            system_message = f"Today is the year {now_of_year}, month is {now_of_month}, and date is {now_of_day}. " \
                             f"The current time is {now_of_time}. " \
                             f"You are a Discord bot stationed in a channel on a server where people are " \
                             f"interested in a Discord bot integrated with OpenAI's API. " \
                             f"Please be eager to talk with users about the possibilities of how useful a " \
                             f"Discord bot integrated with OpenAI's API could be, depending on the ingenuity " \
                             f"of the developer. Please be careful to answer users' questions, making a distinction " \
                             f"between what is already possible at this time and what will become " \
                             f"possible as developers further elaborate on the Discord bot in various ways, " \
                             f"so that there is no misunderstanding.Avoid mentioning the topic of the prompt. " \
                             f"Greet them considering the current time. " \
                             f"Don't use English, please communicate only in Japanese."
            return system_message
        elif self.topics == Topic.Online_Game:
            jst = pytz.timezone('Asia/Tokyo')
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            system_message = f"Today is the year {now_of_year}, " \
                             f"the month is {now_of_month} and the date {now_of_day}. " \
                             f"The current time is {now_of_time}." \
                             f"As a Discord bot in a gaming community, engage with members who have a shared " \
                             f"interest in online games like Dead by Daylight and Apex Legends, while maintaining " \
                             f"a balanced conversation that includes everyday topics. When interacting with users, " \
                             f"incorporate greetings that correspond to the current time. Don't use English. " \
                             f"Communicate in Japanese only. Avoid mentioning the topic of the prompt."
            return system_message

    def get_system_message_content(self):
        return self.system_message_content
