"""Pre-cache OpenCLIP weights and tokenizer on the shared home directory."""

import open_clip


def main():
    model_name = "ViT-B-32"
    pretrained = "openai"

    print(f"Caching OpenCLIP model: {model_name} ({pretrained})")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)

    _ = tokenizer("a photo of a red car")
    print("OpenCLIP model and tokenizer are available.")
    print(f"Loaded model class: {model.__class__.__name__}")


if __name__ == "__main__":
    main()
