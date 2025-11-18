import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "../hugging face/model"   # folder where your trained model was saved

# Load model + tokenizer
print("[DEBUG] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
print("[DEBUG] Model loaded.")


def translate(text: str) -> str:
    """Runs the translator model on input text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=256,
        truncation=True
    )

    output_ids = model.generate(
        **inputs,
        max_length=256,
        num_beams=4
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


# If run with a command-line argument → translate text
if len(sys.argv) > 1:
    input_text = " ".join(sys.argv[1:])
    result = translate(input_text)
    print(result)
else:
    # Interactive mode
    print("Enter text to translate. Press Ctrl+C to exit.\n")
    while True:
        try:
            text = input("> ")
            print(translate(text))
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
