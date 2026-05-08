import segmentation_models_pytorch as smp


def build_model(enocoder_name: str, encoder_weights: str, num_classes: int):
    model = smp.Unet(
        enocoder_name, encoder_weights=encoder_weights, classes=num_classes
    )
    return model
