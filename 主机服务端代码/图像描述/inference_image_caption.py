import open_clip as clip
import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
#from google.colab import files
import skimage.io as io
import PIL.Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Optional[TS]
TA = Union[T, ARRAY]


D = torch.device
CPU = torch.device('cpu')


def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')


CUDA = get_device

# current_directory = os.getcwd()
# save_path = os.path.join(os.path.dirname(current_directory), "pretrained_models")
# os.makedirs(save_path, exist_ok=True)
# model_path = os.path.join(save_path, 'model_wieghts.pt')

class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('./Pretrained/gpt2/')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return




def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
            for i in range(entry_length):
                #这里logits维度为[1,10,50257],其中50257这个维度是词典中每个词的概率,所以logits就代表了每个token对应的词典中每个词的概率
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                #这里只取出一系列token中的最后一个是因为每个token都包含了从自己到前面所有token的信息，所以在预测下一个token的内容时仅需要最后一个token即可
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                #以下几行代码皆是为了找出最后一个token当中对应词典当中最大概率的词的id
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                #这里循环第一次时，next_token为25534，说明对应的是词典中第25534行的那个词向量，也代表它的文本表示是tokenizer中id为25534的文本
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                #我们要利用新预测出的token去继续预测解析来的token，现在我们只有token的id，所以需要经过embedding层找到这个新token所对应的词向量
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    #这里依次记录最新token的id，并与之前的token的id放在一起，方便后面通过tokenizer进行解码，变成我们可以看懂的文本
                    tokens = torch.cat((tokens, next_token), dim=1)
                #将最新预测的token的词向量与之前的结合，并作为输入进入下一次循环。
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break
            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)
    return generated_list[0]

#以下几行使用微调过后的clip
is_gpu = True
device = CUDA(0) if is_gpu else "cpu"
model_name = 'ViT-B-32'
clip_model, _, preprocess = clip.create_model_and_transforms(model_name)
ckpt = torch.load('./Pretrained/best_clip.pt', map_location="cpu")
model_state_dict = ckpt['state_dict']
clip_model.load_state_dict(model_state_dict)

#这里是官方未经过微调的clip，如果要是对遥感图像使用上述clip_model，如果要是针对一般图像，使用下述clip_model_official(无需引入权重，使用官方vit-b-32权重)
clip_model_official= clip.load_openai_model(name='ViT-B-32',device='cpu')

#使用官方训练好的gpt2作为解码器
file = os.getcwd()
file = os.path.join(file,'Pretrained/gpt2/')
tokenizer = GPT2Tokenizer.from_pretrained(file,local_files_only=True)


prefix_length = 10
model = ClipCaptionModel(prefix_length)
model.load_state_dict(torch.load('./Pretrained/conceptual_weights.pt', map_location=CPU),strict=False)
model = model.eval()
device = CUDA(0) if is_gpu else "cpu"
model = model.to(device)

use_beam_search = False #@param {type:"boolean"}

picture_path = './test/test4.jpg'
image = io.imread(picture_path)
pil_image = PIL.Image.fromarray(image)
image_show = mpimg.imread(picture_path)
#pil_img = Image(filename=UPLOADED_FILE)


image = preprocess(pil_image).unsqueeze(0)#.to(device)
with torch.no_grad():
    #这里如果要是针对一般图片使用clip_model_official.encode_image即可
    prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
    #
    prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)
print(generated_text_prefix)


