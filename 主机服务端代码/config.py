CONFIG = {
  'doc_source': './docs/', #需要向量化的文档
  'embedding_model': "shibing624/text2vec-base-chinese", #embeding模型
  'db_source': "./db",  #向量化数据库
  'chunk_size': 200,  #块词量
  'chunk_overlap': 20, #交集范围
  'k': 3,#查询文档量
}