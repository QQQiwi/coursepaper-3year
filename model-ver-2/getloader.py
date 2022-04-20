import os
import pandas as pd
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

# будут определяться слова английского языка
SPACY_ENG = spacy.load('en_core_web_sm')


class Vocabulary:
    """
        Класс словаря, описывающий структуры, содержащие слова, которые запоминает модель
        в процессе обучения и на основе которых формирует подпись к изображению.
    """

    def __init__(self, freq_threshold):
        """
            Функция инициализации, при котором генерируются объекты типа словарь (первый
            объект - определяет слово по индексу, второй - определяет индекс по слову).

        :param freq_threshold: максимальное количество повторяющихся слов
        """

        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        """
            Функция возвращает длину словаря.

        :return: длина словаря (ключ - индекс, значение - слово в словаре)
        """
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        """
            Функция токенизирует слова некоторого текста.

        :param text: входной текст на английском языке
        :return: список токенов для каждого слова в тексте
        """

        # пример: "I love bananas" -> ["i", "love", "bananas"]
        return [tok.text.lower() for tok in SPACY_ENG.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        """
            Функция осуществляет заполнение словаря (добавление слов в него и определение
            для каждого из них индекса).

        :param sentence_list: список предложений, являющихся подписям к изображениям
        """

        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        """
            Функция заменяет слова токенизированного текста соответствующими этим
            словам индексами.

        :param text: некоторый текст, подлежащий токенизированию
        :return: список индексов
        """
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    """
        Определение класса датасета для набора данных Flickr.
    """

    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        """
            Инициализирует объект датасета, определяя списки путей к изображениям и
            соответствующим им подписям, а также формирует словарь.

        :param root_dir: корневая директория, откуда будут браться изображения
        :param caption_file: адрес файла с подписями к изображениям
        :param transform: некоторый объект, преобразовывающий изображения заданным образом
        :param freq_threshold: максимальное количество повторяющихся слов
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocabulary = Vocabulary(freq_threshold)
        self.vocabulary.build_vocabulary(self.captions.tolist())

    def __len__(self):
        """
            Функция возвращает длину датафрейма (т.е. количество различных
            подписей к изображениям).

        :return: длина датафрейма
        """
        return len(self.df)

    def __getitem__(self, index):
        """
            Функция определяет возможность взятия одной сущности из датасета (в частности,
            изображения и соответствующей ему подписи).

        :param index: индекс сущности, которую необходимо считать
        :return: кортеж из трансформированного изображения и тензора преобразованной в
                 индексы слов подписи
        """
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocabulary.stoi["<SOS>"]]
        numericalized_caption += self.vocabulary.numericalize(caption)
        numericalized_caption.append(self.vocabulary.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class MyCollate:
    """
        Класс MyCollate для помещения в batch значений датасета.

    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets


def get_loader(
        root_folder,
        annotation_file,
        transform,
        batch_size=32,
        num_workers=8,
        shuffle=True,
        pin_memory=True
):
    """
    Функция get_loader осуществляет загрузку данных для работы модели машинного
    обучения с этими данными.

    :param root_folder: корневая директория, откуда будут браться изображения
    :param annotation_file: путь к файлу с подписями для изображений
    :param transform: преобразование, которое необходимо сделать с изображениями
    :param batch_size: размер batch
    :param num_workers: определяет количество некоторого ресурса компьютера для ускорения работы
    :param shuffle: параметр перемешивания сущностей
    :param pin_memory: определяет количество некоторого ресурса компьютера

    :return: кортеж из объекта загрузки данных и объекта датасета
    """
    dataset = FlickrDataset(root_folder, annotation_file, transform)
    pad_idx = dataset.vocabulary.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx)
    )

    return loader, dataset


def main():
    """
        Запуск функцию осуществляет проверку корректности созданных классов.
    """
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataloader = get_loader("archive/images", annotation_file="archive/captions.txt",
                            transform=transform)

    for idx, (imgs, captions) in enumerate(dataloader):
        print(imgs.shape)
        print(captions.shape)


if __name__ == "__main__":
    main()
