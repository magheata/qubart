from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Summarizer:

    @staticmethod
    def get_summary(model_name, input_text):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # encode input context
        input_ids = tokenizer(input_text, truncation=True, padding=True, return_tensors="pt").input_ids
        # generate summary
        outputs = model.generate(input_ids=input_ids)
        # decode summary
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
