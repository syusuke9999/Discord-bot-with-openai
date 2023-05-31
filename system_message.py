import datetime
import pytz


class SystemMessage:
    def __init__(self, *args, **kwargs):
        if args is None:
            self.topics = 'None'
            self.system_message_content = 'You are useful assistant.'
        else:
            self.topics = args[0]
            self.system_message_content = self.set_system_message_content()

    def set_system_message_content(self):
        if self.topics is 'discord_bot':
            jst = pytz.timezone('Asia/Tokyo')
            # 現在の日付と時刻を取得
            datetime_jst = datetime.datetime.now(jst)
            now = datetime_jst
            now_of_year = now.strftime("%Y")
            now_of_month = now.strftime("%m")
            now_of_day = now.strftime("%d")
            now_of_time = now.strftime("%H:%M")
            system_message = f'Today is the year {now_of_year}, the month is {now_of_month} ' \
                             f'and the date {now_of_day}. The current time is {now_of_time}.' \
                             f'You are a Discord bot on a channel for people who are interested in a ' \
                             f'Discord bot that works with OpenAI\'s API, and would like to have a ' \
                             f'conversation about the possibilities of a Discord bot that works with ' \
                             f'OpenAI\'s API.Avoid mentioning the topic of the prompt and greet them ' \
                             f'considering the current time.Don\'t use English, ' \
                             f'please communicate only in Japanese.'
            return system_message
        elif self.topics is 'dbd_apex_online_etc':
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
                             f"You are a Discord bot joining a Discord channel where people enjoy " \
                             f"online games. Have fun talking about Dead by Daylight and Apex legend, " \
                             f"and other everyday conversation. with the channel participants." \
                             f"Avoid mentioning the topic of the prompt and greet them considering " \
                             f"the current time." \
                             f"Don't use English, please communicate only in Japanese."
            return system_message

    def get_system_message_content(self):
        return self.system_message_content
