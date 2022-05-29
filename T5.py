import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModelForSeq2SeqLM
from configs import T5_CONFIGS

MAX_LENGTH=256 # tokens

class t5encoder:
    def __init__(self,model_name):

        self.model_name=model_name
        if self.model_name not in T5_CONFIGS.keys():
            print('model name is not found in config')

        self.config = T5_CONFIGS[self.model_name]

        if self.config[0] == 't5':
            t5_class, tokenizer_class = T5EncoderModel, T5Tokenizer

        elif self.config[0] == 'auto':
            t5_class, tokenizer_class = AutoModelForSeq2SeqLM, AutoTokenizer

        else:
            raise ValueError(f'unknown source {config[0]}')

        self.t5_model = t5_class.from_pretrained(model_name)

        self.tokenizer = tokenizer_class.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.t5_model = self.t5_model.cuda()

        self.device = next(self.t5_model.parameters()).device
        print("the device is", self.device)

    def get_encoded_text(self,texts,save_to_file=True):


        encoded = self.tokenizer.batch_encode_plus(texts, return_tensors="pt",
                                              padding='longest',
                                              max_length=MAX_LENGTH,
                                              truncation=True)

        input_ids = encoded.input_ids.to(self.device)
        attn_mask = encoded.attention_mask.to(self.device)

        self.t5_model.eval()

        with torch.no_grad():
            if self.config[0] == 't5':

                output = self.t5_model(input_ids=input_ids, attention_mask=attn_mask)
                encoded_text = output.last_hidden_state
                if save_to_file:
                    encoded_text=encoded_text.detach()
                    attn_mask = attn_mask.detach()

            elif self.config[0] == 'auto':

                output = self.t5_model(input_ids=input_ids, attention_mask=attn_mask, decoder_input_ids=input_ids[:, :1])
                encoded_text = output.encoder_last_hidden_state

                if save_to_file:
                    encoded_text=encoded_text.detach()
                    attn_mask=attn_mask.detach()

        return encoded_text, attn_mask.bool()

    def test(self):
        print("1 sentence test")
        Texts=["I love rock and roll"]
        #model_name='t5-small'
        encoded_text,attention_masks= self.get_encoded_text(Texts,save_to_file=True)
        #print("multi sentence test")
        #Texts=["I love rock and roll",'especially the kind of stuff that rocks']
        #encoded_text, attention_masks = self.get_encoded_text(Texts, model_name)
        #print(encoded_text.size())
        #print(attention_masks.size())
