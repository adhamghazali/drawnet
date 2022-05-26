import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM

T5_CONFIGS = {'t5-small': ['t5',512] ,'t5-base':['t5',768],
              't5-large':['t5',1024] , 'google/t5-v1_1-small': ['auto',512],
              'google/t5-v1_1-base':['auto', 768],
              'google/t5-v1_1-large':['auto',1025]}

MAX_LENGTH=256 # tokens

def get_encoded_text(texts, model_name='t5-small'):
    global T5_CONFIGS
    if model_name not in T5_CONFIGS.keys():
        print('model name is not found in config')

    config = T5_CONFIGS[model_name]

    if config[0] == 't5':
        t5_class, tokenizer_class = T5EncoderModel, T5Tokenizer

    elif config[0] == 'auto':
        t5_class, tokenizer_class = AutoModelForSeq2SeqLM, AutoTokenizer

    else:
        raise ValueError(f'unknown source {config[0]}')

    t5 = t5_class.from_pretrained(model_name)

    tokenizer = tokenizer_class.from_pretrained(model_name)

    if torch.cuda.is_available():
        t5 = t5.cuda()

    device = next(t5.parameters()).device

    encoded = tokenizer.batch_encode_plus(texts, return_tensors="pt",
                                          padding='longest',
                                          max_length=MAX_LENGTH,
                                          truncation=True)

    input_ids = encoded.input_ids.to(device)
    attn_mask = encoded.attention_mask.to(device)

    t5.eval()

    with torch.no_grad():
        if config[0] == 't5':
            output = t5(input_ids=input_ids, attention_mask=attn_mask)
            encoded_text = output.last_hidden_state.detach()

        elif config[0] == 'auto':
            output = t5(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=input_ids[:, :1])
            encoded_text = output.encoder_last_hidden_state.detach()

    return encoded_text, attn_mask.bool()

def test():
    print("1 sentence test")
    Texts=["I love rock and roll"]
    model_name='t5-small'
    encoded_text,attention_masks= get_encoded_text(Texts, model_name)
    print("multi sentence test")

    Texts=["I love rock and roll",'especially the kind of stuff that rocks']
    encoded_text, attention_masks = get_encoded_text(Texts, model_name)


