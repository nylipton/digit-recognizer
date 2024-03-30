import sys
import lightning as L
import torch
from lightning.pytorch.tuner import Tuner
from torchvision import transforms
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from shared_utilities import MnistDataModule, LightningModel, PyTorchCNN, MyProgressBar, save_predictions_to_csv
from watermark import watermark
from albumentations import ElasticTransform

num_epochs = 25
checkpoint = 'MNIST-digit-recognizer.ckpt'
my_model_name = 'MNIST-digit-recognizer-vgg11'
best_lr = 0.13182567385564073 # from Trainer

if __name__ == "__main__":
    print(watermark(packages="torch,lightning", python=True))
    print("Torch CUDA available?", torch.cuda.is_available())

    load = False
    args = sys.argv[1:]
    if len(args) > 0:
        load = True
        checkpoint = args[0]

    pytorch_model = PyTorchCNN(num_classes=10, dropout_rate=0.01)

    if load:
        lightning_model = LightningModel.load_from_checkpoint( checkpoint, )
        dm = MnistDataModule()
        dm.prepare_data()
        dm.setup()
        # lightning_model.eval()
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            callbacks = [MyProgressBar()]
        )
        with torch.no_grad():
            predictions = trainer.predict( model=lightning_model,
                                           dataloaders=dm.predict_dataloader(), )
        save_predictions_to_csv(predictions)
        print('*** predictions saved')
    else:
        # lightning_model = LightningModel.load_from_checkpoint(checkpoint, )
        train_transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.RandomCrop((128, 128)),
                transforms.RandomRotation(degrees=10),
                # ElasticTransform(alpha=120, sigma=120, alpha_affine=120, ),
                transforms.Resize( (24, 24 ) ),
                transforms.ToTensor(),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((150, 150)),
                transforms.CenterCrop((128, 128)),
                transforms.Resize((24, 24)),
                transforms.ToTensor(),
            ]
        )
        dm = MnistDataModule( train_transform=train_transform, test_transform = test_transform )
        dm.prepare_data()
        dm.setup(stage='train')
        num_steps = num_epochs * len(dm.train_dataloader())  # # epochs * # minibatches
        lightning_model = LightningModel(model=pytorch_model, learning_rate=best_lr, cos_t_max=num_steps)
        trainer = L.Trainer(
            max_epochs=num_epochs,
            accelerator="cpu",
            devices=1,
            # overfit_batches=1,
            logger=[CSVLogger(save_dir="logs/", name=my_model_name),
                    TensorBoardLogger(save_dir="~/Desktop/lightning_logs/digit-recognizer")],
            # deterministic=True,
            callbacks=[MyProgressBar(),  # this fixes a PyCharm bug with output
                       ModelCheckpoint(save_top_k=2, mode="max", monitor="val_acc", save_last=True)],
        )
        # tuner = Tuner(trainer)
        # lr_finder = tuner.lr_find(lightning_model, datamodule=dm)
        # lr_finder.plot(suggest=True)
        trainer.fit(model=lightning_model, datamodule=dm, )
        trainer.save_checkpoint( f'{my_model_name}.ckpt')

        # now make predictions
        with torch.no_grad():
            dm.setup(stage='test')
            predictions = trainer.predict( model=lightning_model,
                                           dataloaders=dm.predict_dataloader(), )
            print( '*** DONE WITH PREDICTIONS')
            save_predictions_to_csv( predictions )
