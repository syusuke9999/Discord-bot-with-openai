from langchain.callbacks.manager import AsyncCallbackManagerForChainRun, CallbackManagerForChainRun
from langchain.chains.retrieval_qa.base import RetrievalQA as BaseRetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List
from datetime import datetime
import pytz


class CustomRetrievalQA(BaseRetrievalQA):

    message_history: List[str] = []

    def __init__(self, model, tokenizer, preprocessor, postprocessor, evaluator, trainer, predictor, dataset,
                 data, config, metrics, losses, optimizers, schedulers, callbacks, logs):
        super().__init__(model=model, tokenizer=tokenizer, preprocessor=preprocessor, postprocessor=postprocessor,
                         evaluator=evaluator, trainer=trainer, predictor=predictor, dataset=dataset,
                         data=data, config=config, metrics=metrics, losses=losses, optimizers=optimizers,
                         schedulers=schedulers, callbacks=callbacks, logs=logs)
        # カスタムプロンプトの定義
        # 現在の日付と時刻を取得します（日本時間）。
        now = datetime.now(pytz.timezone('Asia/Tokyo'))
        # 年、月、日を取得します。
        year = now.year
        month = now.month
        day = now.day
        custom_prompt = ("Previous Conversation: {message_histories}\n"
                         f"Today is the year {year}, the month is {month} and the date {day}."
                         f"The current time is {now}."
                         "Use the following pieces of context to answer the question at the end. If you don't "
                         "know the answer,"
                         " just say 「分かりません」, don't try to make up an answer. "
                         "Please use Japanese only. Don't use English."
                         " \n"
                         "Context:{context}"
                         " \n"
                         "Question: {question}"
                         "Helpful Answer:")
        self.prompt_template = PromptTemplate(
            template=custom_prompt,
            input_variables=["message_history", "context", "question"]
        )

    def add_message_to_history(self, message: str):
        self.message_history.append(message)

    async def _aget_docs(
            self,
            question: str,
            *,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        # メッセージの履歴を考慮したロジック
        modified_question = self._modify_question_based_on_history(question)
        return self.retriever.get_relevant_documents(modified_question)

    def _get_docs(
            self,
            question: str,
            *,
            run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        # メッセージの履歴を考慮したロジック
        modified_question = self._modify_question_based_on_history(question)
        return self.retriever.get_relevant_documents(modified_question)

    def _modify_question_based_on_history(self, question: str) -> str:
        # 履歴に基づいて質問を変更するロジック
        return " ".join(self.message_history) + " " + question

    def _generate_prompt(self, context: str, question: str) -> str:
        # カスタムプロンプトを使用してプロンプトを生成
        return self.prompt_template.format(
            message_history=" ".join(self.message_history),
            context=context,
            question=question
        )
