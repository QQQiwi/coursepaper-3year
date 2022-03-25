import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import tensorboard
from torch.utils.tensorboard import SummaryWriter
# from utils import save_checkpoint, load_checkpoint, print_examples
from model import CNNtoRNNTranslator
from get_loader import get_loader


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="archive/images",
        annotation_file="archive/training_captions.txt",
        transform=transform,
        num_workers=6,
    )

    torch.backends.cudnn.benchmark = True
    print("cuda: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    # load_model = True
    # save_model = False

    # Гиперпараметры
    embedding_size = 256
    hidden_size = 256
    vocabulary_size = len(dataset.vocabulary)
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 100

    # для tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # инициализация моделей, функции потерь и т.д.
    model = CNNtoRNNTranslator(embedding_size, hidden_size, vocabulary_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocabulary.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        checkpoint = torch.load("my_checkpoint.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]

    model.train()

    for epoch in range(num_epochs + 1):
        print("Epoch: ", epoch)

        if save_model and (epoch % 10 == 0):
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            torch.save(checkpoint, f"my_checkpoint_{epoch}.pth.tar")

        for idx, (imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()


if __name__ == "__main__":
    train()
