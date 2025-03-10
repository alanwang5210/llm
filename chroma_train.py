import chromadb

# 获取Chroma客户端，并创建集合（Collection，类似于关系型数据库的一张表）。
# Collection负责存储向量、文档和其他元数据。
# chroma_client = chromadb.Client()
# 将数据持久化存储到本地磁盘，可以将本地磁盘中的目录地址传递给Chroma
chroma_client = chromadb.PersistentClient(path="./chroma")

# create_collection()方法还带有可选的参数metadata，通过设置其值可以自定义向量空间的距离计算方法。
# 此处设置距离计算方法是余弦相似度。hnsw:space的有效选项包括l2（欧氏距离）、ip（点积）和consine（余弦相似度）。
# collection = chroma_client.create_collection(
#     name="my_collection",
#     metadata={"hnsw:space": "cosine"})
#
collection = chroma_client.get_collection(name="my_collection")

# 接下来向Chroma中添加数据。如果向Chroma传递了文本信息列表，则程序会将其嵌入Collection的Embedding()方法中，
# 以便将文本信息转化为词向量。当文本信息过大而无法使用所选的Embedding()方法时，会产生异常。
# Embedding（词向量或词嵌入）是集合的重要组成部分。可以根据Chroma内部包含的Embedding模型隐式生成，
# 或者基于OpenAI公司等提供的外部模型生成Embedding。Chroma默认使用Sentence Transformers
# 的all-MiniLM-L6-v2模型创建Embedding。在本地运行时，需要下载模型文件

# 在添加数据时，每个数据必须有一个唯一的关联ID。在下面代码中，metadata可以为每个数据提供可选的字典列表，
# 以存储附加信息并在过滤操作中发挥作用。在这个示例中，每条数据都关联一个包含单个键“source”和对应值“my_source”的字典。
# 其中，ids参数指定每条数据的唯一标识符列表。在Chroma中，每条数据都可以有一个与之关联的唯一ID，用于后续对该文档的引用和检索。
collection.add(documents=["This is a document", "This is another document"],
               metadatas=[{"source": "my_source"}, {"source": "my_source"}],
               ids=["id1", "id2"])

# 如果已经生成了词向量，也可以直接将其添加到Collection中。
# collection.add(embeddings=[[1.2, 2.3, 4.5], [6.7, 8.2, 9.2]],
#                documents=["This is a document", "This is another document"],
#                metadatas=[{"source": "my_source"}, {"source": "my_source"}],
#                ids=["id1", "id2"])


# 使用query()方法来查询Collection中与查询文本（如下示例中的query_texts）最相近的n个结果
# results = collection.query(query_texts=["This is a query document"],
#                            n_results=2)

# 通过设置参数query_embedding（传入的是向量而不是文本）来进行查询。同时，还可以通过可选的参数where和where_document进行过滤
# results = collection.query(
#     query_embeddings=[[11.1, 12.1, 13.1], [1.1, 2.3, 3.2]],
#     n_results=10,
#     where={"metadata_field": "is_equal_to_this"},
#     where_document={"$contains": "search_string"}
# )

# Chroma向量数据库支持基于元数据或ids的查询。在下面的查询代码中，首先执行相似性搜索，然后根据参数where过滤查询，该参数指定了具体的元数据。
results = collection.query(
    query_texts=["another"],
    where={"source": "my_source"},
    n_results=1
)
print(results)

# 使用delete()方法删除集合中的元素，如删除ids值为“id2”的元素
collection.delete(["id2"])
