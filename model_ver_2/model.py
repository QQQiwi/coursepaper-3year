import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
        Кодировщик, определяющийся с помощью предобученной GoogleNetv3.
    """

    def __init__(self, embedding_size, train_CNN=False):
        """
            Инициализация модели с определением её слоев.

        :param embedding_size: размер эмбеддинга
        :param train_CNN: задается ли CNN для обучения
        """
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embedding_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        """
            Функция, определяющая процесс передачи весов внутри кодировщика.
        
        :param images: список тензоров, полученных преобразованием изображений
        :return: веса модели, пропущенные через relu с осуществлением сброса весов
        """
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    """
        Декодировщик, определяемый с помощью LSTM-модели.
    """

    def __init__(self, embedding_size, hidden_size, vocabulary_size, layers_num):
        """
            Инициализация модели с определением её слоев.

        :param embedding_size: размер эмбеддинга
        :param hidden_size: выходной вектор значений LSTM
        :param vocabulary_size: размер словаря
        :param layers_num: количество слоев, составляющих LSTM
        """
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, layers_num)
        self.linear = nn.Linear(hidden_size, vocabulary_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        """
            Функция определяющая процесс передачи весов внутри декодировщика

        :param features: получаемые вектора, являющиеся результатом работы кодировщика
        :param captions: подписи к изображениям
        :return: подпись к входному изображению
        """
        embeddings = self.dropout(self.embedding(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNNTranslator(nn.Module):
    """
        Класс, описывающий ансамбль, составленный из кодировщика и декодировщика
    """

    def __init__(self, embedding_size, hidden_size, vocabulary_size, layers_num):
        """
            Инициализация элементов ансамбля.

        :param embedding_size: размер эмбеддинга
        :param hidden_size: количество выходных векторов LSTM
        :param vocabulary_size: размер словаря
        :param layers_num: количество слоев, составляющих LSTM
        """
        super(CNNtoRNNTranslator, self).__init__()
        self.encoderCNN = EncoderCNN(embedding_size)
        self.decoderRNN = DecoderRNN(embedding_size, hidden_size, vocabulary_size, layers_num)

    def forward(self, images, captions):
        """
            Процесс передачи весов между элементами ансамбля

        :param images: тензоры изображений
        :param captions: подписи к изображениям
        :return: сгенерированные подписи к изображениям
        """
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def image_caption(self, image, vocabulary, max_length=50):
        """
            Функция генерирует подпись (максимальная длина - 50 символов) к изображению на основе словаря,
            предварительно обработав входное изображение.

        :param image: изображение
        :param vocabulary: словарь
        :param max_length: максимальная длина генерируемой подписи
        :return: вектор слов, определяющий подпись к изображению
        """
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoderRNN.embedding(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

            return [vocabulary.itos[idx] for idx in result_caption]
