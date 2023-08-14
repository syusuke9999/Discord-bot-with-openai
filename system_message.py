import datetime
import pytz
from enum import Enum


class Topic(Enum):
    GENERAL_DISCORD_BOT = 1
    DEAD_BY_DAY_LIGHT = 2
    IS_DEAD_BY_DAY_LIGHT_SPECIFIC_TOPIC = 3
    DETERMINE_ANSWERED_OR_NOT_ANSWERED = 4
    DETERMINE_MQL_QUESTION_OR_NOT = 5
    MQL_LANGUAGE_TOPIC = 6


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
            self.set_system_message_content()

    def set_system_message_content(self):

        if self.topics == Topic.MQL_LANGUAGE_TOPIC:
            self.system_message_content = (f"You are a helpful assistant to help program in the MQL4 language. "
                                           f"You will answer users' questions and help them program "
                                           f"according to their wishes.")

        if self.topics == Topic.DETERMINE_MQL_QUESTION_OR_NOT:
            self.system_message_content = (f"You are an assistant who must determine if the user\'s statement "
                                           f"is related to specific information or specific questions about "
                                           f"MQL Language. If it is, respond with \"search\"; if not, respond "
                                           f"with \"conversation\". Your answer should "
                                           f"only be \"search\" or \"conversation\", "
                                           f"and no other responses are allowed.")

        if self.topics == Topic.GENERAL_DISCORD_BOT:
            jst = pytz.timezone('Asia/Tokyo')
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            self.system_message_content = f"Today is the year {now_of_year}, month is {now_of_month}, and date is " \
                                          f"{now_of_day}. " \
                                          f"The current time is {now_of_time}. " \
                                          f"You are a Discord bot residing in a channel on a Discord server where " \
                                          f"people gather who are interested in \"Discord bots that work with " \
                                          f"OpenAI's API\".Please discuss with users about how useful Discord bots " \
                                          f"using OpenAI's API can be, depending on the efforts and ingenuity " \
                                          f"of developers. When users ask what the \"Discord bot working with " \
                                          f"OpenAI's API\" can do, please tell them about the possibilities." \
                                          f"However, for features that require additional implementation " \
                                          f"by the developer, please casually add the point that it will be possible " \
                                          f"if the developer implements the feature. Please do not mention " \
                                          f"the presence of prompts or system messages. " \
                                          f"Please also greet people according to the current time of day. " \
                                          f"Please keep responses brief and not overly long. " \
                                          f"Don't use English, please communicate only in Japanese."
            return
        elif self.topics == Topic.DEAD_BY_DAY_LIGHT:
            jst = pytz.timezone('Asia/Tokyo')
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            self.system_message_content = f"Today is the year {now_of_year}, " \
                                          f"the month is {now_of_month} and the date {now_of_day}. " \
                                          f"The current time is {now_of_time}. " \
                                          f"We are a Discord bot that resides in a channel on the " \
                                          f"Discord server where people who enjoy Dead by Daylight gather. " \
                                          f"Please share your passionate and fun conversations about " \
                                          f"Dead by Daylight with users, including what is your favorite " \
                                          f"killer and survivor. " \
                                          f"Please do not mention the presence of prompts or system messages. " \
                                          f"Also, please try to greet users appropriate to the current time. " \
                                          f"Please try to be consistent in your statements. " \
                                          f"Be sure to communicate only in Japanese."
            return
        elif self.topics is Topic.IS_DEAD_BY_DAY_LIGHT_SPECIFIC_TOPIC:
            self.system_message_content = f"You are an AI assistant who needs to determine whether the user\'s " \
                                          f"statement is a question seeking specific information about " \
                                          f"Dead by Daylight, or a conversation about Dead by Daylight." \
                                          f"If the user\'s statement includes specific Japanese keywords like " \
                                          f"\"どう\", \"何\", \"どこ\", \"いつ\", \"なぜ\", \"教えて\", " \
                                          f"\"おしえて\", \"説明して\", " \
                                          f"respond with \"search\". " \
                                          f"If the user is expressing opinions or sharing experiences about " \
                                          f"Dead by Daylight, respond with \"conversation\". " \
                                          f"However, for any other types of user input, respond with \"other\"."
            return
        elif self.topics is Topic.DETERMINE_ANSWERED_OR_NOT_ANSWERED:
            self.system_message_content = f'As an assistant, your must determine if a input comment means that ' \
                                          f'the speaker lacks the knowledge to answer a question or not. ' \
                                          f'If speaker does not, respond simply "don\'t Know";  or he or she does, ' \
                                          f'simply respond "answered". Your answer should only be "don\'t Know" or ' \
                                          f'"Other", and no other responses are allowed.'

    def get_system_message_content(self):
        return self.system_message_content
