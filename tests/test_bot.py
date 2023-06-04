import unittest
from unittest.mock import Mock
from main import MyBot


class TestMyBot(unittest.TestCase):
    def setUp(self):
        self.bot = MyBot(command_prefix='!', intents=None, enum_of_topic=None)

    def test_remove_duplicate_messages(self):
        self.bot.message_histories = {
            'user1': [
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'},
                {'role': 'user', 'content': 'Hello'},
                {'role': 'assistant', 'content': 'Hi'},
            ]
        }
        self.bot.remove_duplicate_messages('user1', {'role': 'user', 'content': 'Hello'})
        self.assertEqual(len(self.bot.message_histories['user1']), 2)


if __name__ == '__main__':
    unittest.main()
