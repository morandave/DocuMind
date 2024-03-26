import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
import requests
from config import CONFIG
import subprocess
embedding_model = CONFIG['embedding_model']

# 用户上传文件，将所有文件存入一个文件夹（文件夹名称待定）
input_dir = "c++引用的好处"


# 将文件夹地址作为参数传入createKnowledgeBase.py，产生知识库（知识库名称已知）
# subprocess.run(["python3", "testCreateKnowledgeBase.py", "--input_dir", input_dir])
output_dir = output_dir = os.path.join(CONFIG['db_source'], input_dir)# 知识库文件夹


# 用户输入问题
question = "C++中引用的注意事项有哪些？"



# 根据问题从知识库中寻找最相近知识，组成prompt
embedding_function = SentenceTransformerEmbeddings(model_name=embedding_model)
db = Chroma(persist_directory=output_dir, embedding_function=embedding_function)
knowledge = db.similarity_search(question, k=3)  # 摘取数量
    
#合并检索结果
merge_message = ''
for c in knowledge:
    merge_message += c.page_content
merge_message = merge_message
    
prompt_template = f"""基于以下已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，
请说 "根据已知信息无法回答该问题" 或 "没有提供足够的相关信息"，不允许在答案中添加编造成分，答案请使用中文。
已知内容:
------------------------------------------------------------------------------
    {merge_message}
------------------------------------------------------------------------------
问题:
*************************
    {question}
*************************
"""
    
data = {
    "prompt": prompt_template
}

# 发送POST请求
response = requests.post("http://192.168.55.8:8080", json=data)

# 获取响应结果
result = response.json()

# 打印结果
print(result)
