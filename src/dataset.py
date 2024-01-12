from imports import *
from constant import *

class ClimateTextDataset(Dataset):
    def __init__(self, X ,y ,tokenizer, max_token_len=MAX_LENGHT):
            self.X = X
            self.tokenizer = tokenizer
            self.max_token_len = max_token_len
            self.y = y

    def __len__(self):
            return len(self.X)

    def __getitem__(self, idx):
            try:
                    extracted_text = self.X[idx]
                    label = self.y[idx]

                    encodings = self.tokenizer.encode_plus(
                            text = extracted_text,
                            add_special_tokens = True,
                            max_length = self.max_token_len,
                            return_token_type_ids = False,
                            padding="max_length",
                            truncation=True,
                            return_attention_mask = True,
                            return_tensors='pt',
                            # is_split_into_words=True
                    )

                    return dict(
                            text = extracted_text,
                            input_ids = encodings['input_ids'].flatten(),
                            attention_mask = encodings['attention_mask'].flatten(),
                            label = torch.tensor(float(label)))
            except:
                    extracted_text = self.X[idx]
                    print(extracted_text)
                    print(f'failed at {idx}')


class ClimateTextDatasetTestPhase(Dataset):
        def __init__(self, X, tokenizer, max_token_len=MAX_LENGHT):
                self.X = X
                self.tokenizer = tokenizer
                self.max_token_len = max_token_len

        def __len__(self):
                return len(self.X)

        def __getitem__(self, idx):
                try:
                        extracted_text = self.X[idx]

                        encodings = self.tokenizer.encode_plus(
                                text = extracted_text,
                                add_special_tokens = True,
                                max_length = self.max_token_len,
                                return_token_type_ids = False,
                                padding="max_length",
                                truncation=True,
                                return_attention_mask = True,
                                return_tensors='pt',
                                # is_split_into_words=True
                        )

                        return dict(
                                text = extracted_text,
                                input_ids = encodings['input_ids'].flatten(),
                                attention_mask = encodings['attention_mask'].flatten(),
                        )
                except:
                        extracted_text = self.X[idx]
