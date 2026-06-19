
import argparse
import re
from pathlib import Path

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer


# I download the NLTK resources needed for sentence tokenisation.
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# I apply the same light text cleaning used in the notebook pipeline.
def clean_text(text):
    text = str(text).replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# I read a text file with flexible encoding support.
def read_text_file(path):
    for enc in ["utf-8", "utf-8-sig", "latin1", "gb18030", "gbk", "utf-16"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read(), enc
        except Exception:
            pass
    raise ValueError(f"Could not read file: {path}")


# I implement the Lead-1 baseline by returning the first sentence.
def lead1(article):
    sents = nltk.sent_tokenize(article)
    return sents[0].strip() if len(sents) > 0 else ""


# I implement the TextRank baseline by selecting one ranked sentence.
def textrank(article):
    article = str(article).strip()
    if article == "":
        return ""
    try:
        parser = PlaintextParser.from_string(article, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        sents = summarizer(parser.document, 1)
        return " ".join(str(s).strip() for s in sents).strip()
    except Exception:
        return ""


# I load a sequence-to-sequence model and its tokenizer for inference.
def load_seq2seq(model_name_or_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path).to(device)
    model.eval()
    return tokenizer, model


# I generate one summary from a sequence-to-sequence model.
def seq2seq_summary(article, tokenizer, model, generation_kwargs, device, max_input_length):
    inputs = tokenizer(
        [article],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)

    return tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0].strip()


# I build the CLI argument parser with two input modes.
def parser_builder():
    parser = argparse.ArgumentParser(description="Generate summaries with the project summarisation methods.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw news article text.")
    group.add_argument("--file_path", type=str, help="Path to a text file.")
    return parser


# I run the full CLI workflow from input reading to summary generation.
def main():
    setup_nltk()
    parser = parser_builder()
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    finetuned_dir = project_dir / "finetuned_bart_xsum" / "model"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # I accept either direct text input or a text file path.
    if args.text is not None:
        article = args.text
        input_source = "raw_text_argument"
    else:
        article, enc = read_text_file(args.file_path)
        input_source = f"file_path ({args.file_path}, encoding={enc})"

    # I clean the input article and stop if nothing remains.
    article = clean_text(article)
    if article == "":
        raise ValueError("The input article is empty after cleaning.")

    print("=" * 100)
    print("News Summarisation CLI Demo")
    print("=" * 100)
    print(f"Input source: {input_source}")
    print(f"Device: {device}")
    print(f"Article word count: {len(article.split())}")
    print("=" * 100)

    # I generate summaries from the baseline methods.
    lead1_summary = lead1(article)
    textrank_summary = textrank(article)

    # I load and run the DistilBART XSum model.
    tok_xsum, mdl_xsum = load_seq2seq("sshleifer/distilbart-xsum-12-6", device)
    distilbart_xsum_summary = seq2seq_summary(
        article, tok_xsum, mdl_xsum,
        {
            "num_beams": 4,
            "max_length": 40,
            "min_length": 5,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        },
        device,
        1024
    )

    # I load and run the DistilBART CNN/DailyMail model.
    tok_cnn, mdl_cnn = load_seq2seq("sshleifer/distilbart-cnn-12-6", device)
    distilbart_cnn_summary = seq2seq_summary(
        article, tok_cnn, mdl_cnn,
        {
            "num_beams": 4,
            "max_length": 80,
            "min_length": 10,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 3,
            "early_stopping": True
        },
        device,
        1024
    )

    # I load and run the fine-tuned BART model if the saved model directory is available.
    if finetuned_dir.exists():
        tok_ft, mdl_ft = load_seq2seq(str(finetuned_dir), device)
        finetuned_summary = seq2seq_summary(
            article, tok_ft, mdl_ft,
            {
                "num_beams": 4,
                "max_length": 60,
                "min_length": 5,
                "length_penalty": 1.0,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            },
            device,
            512
        )
    else:
        finetuned_summary = "[Fine-tuned model directory not found.]"

    # I collect all summary outputs in one list for printing.
    outputs = [
        ("Lead-1", lead1_summary),
        ("TextRank", textrank_summary),
        ("distilbart-xsum-12-6", distilbart_xsum_summary),
        ("distilbart-cnn-12-6", distilbart_cnn_summary),
        ("facebook/bart-base fine-tuned on XSum", finetuned_summary),
    ]

    # I print the summary from each method in a clear section.
    for name, summary in outputs:
        print("\n" + "=" * 100)
        print(name)
        print("=" * 100)
        print(summary)

    print("\n" + "=" * 100)
    print("CLI run complete.")
    print("=" * 100)


if __name__ == "__main__":
    main()
