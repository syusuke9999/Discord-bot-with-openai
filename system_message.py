import datetime
import pytz


class SystemMessage:
    def __init__(self):
        jst = pytz.timezone('Asia/Tokyo')
        # 現在の日付と時刻を取得
        datetime_jst = datetime.datetime.now(jst)
        now = datetime_jst
        now_of_year = now.strftime("%Y")
        now_of_month = now.strftime("%m")
        now_of_day = now.strftime("%d")
        now_of_time = now.strftime("%H:%M")
        self.system_message_content = f"Today is the year {now_of_year}, the month is {now_of_month} " \
                                      f"and the date {now_of_day}. The current time is {now_of_time}." \
                                      f"You are a Discord bot on a channel for people who are interested in a " \
                                      f"Discord bot that works with OpenAI's API, and would like to have a " \
                                      f"conversation about the possibilities of a Discord bot that works with " \
                                      f"OpenAI's API.Avoid mentioning the topic of the prompt and greet them " \
                                      f"considering the current time.Don't use English, " \
                                      f"please communicate only in Japanese."

    def get_system_message_content(self):
        return self.system_message_content
