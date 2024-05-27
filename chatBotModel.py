from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from answers import answer
class ChatbotModel:
    def __init__(self):
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.predefined_responses = {
            "привет": "Привет! Как я могу помочь вам сегодня?",
            "как дела": "У меня все хорошо, спасибо! Как у вас?",
            "что ты умеешь": "Я могу отвечать на ваши вопросы и помогать вам с информацией.",
            "Хочу узнать информацию о банке": answer[0],
            "Где находиться банк": answer[0],
            "Как перевести деньги": answer[1],
            "Как скинуть деньги другому человеку": answer[1],
            "Скинуть деньги человеку": answer[1],
            "Как сделать перевод": answer[1],
            "Как изменить паспортные данные": answer[2],
            "Как узнать свои долги": answer[3],
            "Как я могу открыть банковский счет": answer[4],
            "Как я могу узнать баланс своего счета": answer[5],
            "узнать баланс счета": answer[5],
            "Как подать заявку на кредитную карту": answer[6],
            "Какие у вас процентные ставки по вкладам": answer[7],
            "процентные ставки по вкладам": answer[7],
            "Что делать, если банкомат не выдал деньги": answer[8],
            "банкомат не выдал деньги": answer[8],
            "Что делать, если моя карта заблокирована": answer[9],
            "моя карта заблокирована": answer[9],
            "Как узнать остаток по кредиту": answer[10],
            "информация про остаток по кредиту": answer[10],
            "Как сменить PIN-код на карте": answer[11],
            "Как сменить пин код на карте": answer[11],
            "Каковы часы работы ваших отделений": answer[12],
            "до скольки работаете": answer[12],
            "помощь": "Чем я могу вам помочь"

        }
        self.response_embeddings = self.compute_embeddings(list(self.predefined_responses.keys()))

    def compute_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
        return embeddings

    def generate_response(self, prompt):
        prompt_embedding = self.compute_embeddings([prompt])
        similarities = cosine_similarity(prompt_embedding, self.response_embeddings)
        best_match_idx = similarities.argmax()
        best_match_text = list(self.predefined_responses.keys())[best_match_idx]
        return self.predefined_responses[best_match_text]