import torch
from model import *
from getloader import get_loader
import torchvision.transforms as transforms
from PIL import Image
from IPython import display
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np


def load_model_to_captioning(image_path="archive/images",
                             caption_path="archive/captions.txt",
                             checkpoint_path="my_checkpoint.pth.tar"):
    """
        Функция загружает веса модели согласно указанным в аргументах
        директориям, создает объекты загрузчика данных.

    :param image_path: путь к изображениям
    :param caption_path: путь к csv-файлу с текстовыми описаниями для
        изображений
    :param checkpoint_path: путь к файлу весов модели

    :return: кортеж из объекта модели, словаря набора данных, объекта
        преобразований изображения
    """
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    
    test_loader, dataset = get_loader(
        root_folder=image_path,
        annotation_file=caption_path,
        transform=transform,
        num_workers=6,
    )
    
    device = "cuda"
    embedding_size = 256
    hidden_size = 256
    vocabulary_size = len(dataset.vocabulary)
    num_layers = 2
    model = CNNtoRNNTranslator(embedding_size, hidden_size,
                               vocabulary_size, num_layers).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    
    return model, dataset.vocabulary, transform


def create_caption(image_path, model=None, vocabulary=None,
                   transform=None, device="cuda", show_image=False):
    """
        Функция генерирует текстовое описание для изображения.

    :param image_path: путь к изображению
    :param model: файл модели, которая будет генерировать текстовое описание
    :param vocabulary: словарь, слова из которого будут использоваться в
        описании
    :param transform: объект преобразований изображения
    :param device: указание строкой ресурса, с помощью которого будет работать
        модель (cuda или cpu)
    :param show_image: выводить ли изображение на экран перед возвратом функцией
        текстового описания

    :return: кортеж из объекта модели, словаря набора данных, объекта
            преобразований изображения
    """
    if not model:
        capt_path = "archive/training_captions.txt"
        checkp_path = "my_checkpoint_100.pth.tar"
        model, vocabulary, transform = load_model_to_captioning(caption_path=capt_path,
                                                                checkpoint_path=checkp_path)
    model.eval()
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    word_list = model.image_caption(image.to(device), vocabulary)
    word_list = [word for word in word_list if not (word in [*vocabulary.stoi][:4])]
    caption = ' '.join([str(elem) for elem in word_list])
    if show_image:
        display.display(Image.open(image_path))
    return caption


def cosine_similarity(sentence_embeddings, ind_a, ind_b):
    """
        Функция реализует косинусный коэффициент.

    :param sentence_embeddings: список из эмбеддингов сгенерированного и
        истинного текстового описания
    :param ind_a: индекс в этом списке сгенерированного описания
    :param ind_b: индекс в этом списке истинного описания

    :return: действительное значение косинусного коэффициента
    """
    s = sentence_embeddings
    return np.dot(s[ind_a], s[ind_b]) / (np.linalg.norm(s[ind_a])
                                         * np.linalg.norm(s[ind_b]))


def check_quality(images_path="archive/images", captions_path="archive/captions.txt"):
    """
        Функция находит среднее значение косинусного коэффициента относительно
        сгенерированных с помощью модели текстовых описаний и истинных описаний
        тестовой выборки. Выводит это значение, затем выводит интерпретированное
        описание смысла этого значения.

    :param images_path: путь к изображениям
    :param captions_path: путь к csv-файлу с истинными текстовыми описаниями к
        изображениям
    """
    
    df = pd.read_csv(captions_path)
    quality = 0
    sent_trans_model = SentenceTransformer('bert-base-nli-mean-tokens')
    capt_path = "archive/training_captions.txt"
    checkp_path = "my_checkpoint_100.pth.tar"
    model, vocab, transform = load_model_to_captioning(caption_path=capt_path,
                                                       checkpoint_path=checkp_path)

    for elem in df.iterrows():
        image_name = elem[1][1]
        ans_caption = elem[1][2]
        out_caption = create_caption(images_path + "/" + str(image_name),
                                     model,
                                     vocab,
                                     transform)
        
        sentences = [out_caption, ans_caption]
        sentence_embeddings = sent_trans_model.encode(sentences)
        quality += cosine_similarity(sentence_embeddings, 0, 1)

    mean_quality = quality / df.shape[0]
    print("Средняя оценка качества модели: ", mean_quality)

    if mean_quality <= 0.25:
        print("Модель работает плохо, не способна даже в общих чертах описать то,"
              " что присутствует на изображении.")
    elif 0.25 < mean_quality <= 0.5:
        print("Модель работает удовлетворительно - способна различать и корректно"
              " интерпретировать основные детали изображения, но может допускать"
              " некоторые логические и смысловые ошибки в описании.")
    elif 0.5 < mean_quality <= 0.75:
        print("Модель работает хорошо - почти всегда корректно описывает основные"
              " детали изображения, но способна допускать незначительные логические"
              " ошибки.")
    else:
        print("Модель работает отлично, количество ошибок при генерации текстового"
              " описания к входному изображению сведено к минимуму.")



