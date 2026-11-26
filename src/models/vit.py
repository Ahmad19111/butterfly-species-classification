"""ViT mit neuem Klassifikationskopf und optionalem LoRA."""
import timm


def build_vit(model_name, n_classes, pretrained=True):
    # timm initialisiert den Klassifikationskopf für die Zielklassen neu.
    return timm.create_model(model_name, pretrained=pretrained, num_classes=n_classes)


def freeze_backbone(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def extract_features(model, x):
    # Features wie beim Klassifikationskopf vor dem letzten Layer.
    return model.forward_head(model.forward_features(x), pre_logits=True)


def add_lora(model, rank, alpha, dropout, target_modules, modules_to_save):
    from peft import LoraConfig, get_peft_model  # Nur für LoRA erforderlich.

    # PEFT friert die übrigen Gewichte ein.
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                     target_modules=target_modules, modules_to_save=modules_to_save)
    return get_peft_model(model, cfg)


def load_lora(model, ckpt_dir):
    from peft import PeftModel

    # LoRA-Gewichte und gespeicherten Kopf laden.
    return PeftModel.from_pretrained(model, ckpt_dir)
