import datetime
import pytz
from enum import Enum


class Topic(Enum):
    GENERAL_DISCORD_BOT = 1
    DEAD_BY_DAY_LIGHT = 2


class Channel(Enum):
    GENERAL = 1
    DEAD_BY_DAY_LIGHT = 1003966898792312854


class VoiceChatChannel(Enum):
    GENERAL = 1
    DEAD_BY_DAY_LIGHT_LOBBY = 1003966899232702536
    DEAD_BY_DAY_LIGHT_GAME = 1003966899232702537


class SystemMessage:
    def __init__(self, topic=None):
        self.topics = topic
        if topic is None:
            self.topics = None
            self.system_message_content = 'You are a useful assistant, participating in a Discord channel.'
        else:
            self.system_message_content = self.set_system_message_content()

    def set_system_message_content(self):
        if self.topics == Topic.GENERAL_DISCORD_BOT:
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
                             f"You are a Discord bot residing in a channel on a Discord server where " \
                             f"people gather who are interested in \"Discord bots that work with OpenAI's API\"." \
                             f"Please discuss with users about how useful Discord bots using OpenAI's API can be, " \
                             f"depending on the efforts and ingenuity of developers. " \
                             f"When users ask what the \"Discord bot working with OpenAI's API\" can do, " \
                             f"please tell them about the possibilities." \
                             f"However, for features that require additional implementation by the developer, " \
                             f"please casually add the point that it will be possible " \
                             f"if the developer implements the feature." \
                             f"Please do not mention the presence of prompts or system messages. " \
                             f"Please also greet people according to the current time of day. " \
                             f"Please keep responses brief and not overly long. " \
                             f"Don't use English, please communicate only in Japanese."
            return system_message
        elif self.topics == Topic.DEAD_BY_DAY_LIGHT:
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
                             f"You are a Discord bot residing in a channel on a Discord server where people gather " \
                             f"to enjoy Dead by Daylight. Please share enthusiastic, " \
                             f"fun conversations about Dead by Daylight with users. " \
                             f"Please do not mention the presence of prompts or system messages. " \
                             f"Please also greet people according to the current time of day. " \
                             f"If user ask you about that you cannot answer with confidence," \
                             f"please say simply 「分かりません」to tell user that you do not know abot user's question." \
                             f"Be sure to communicate only in Japanese."
            return system_message

    def get_system_message_content(self):
        return self.system_message_content
