"""命令行入口：增量更新课程知识库向量索引。"""

from langchain_community.vectorstores import FAISS

from app.rag.vectorstore import ingest_indexes


def ingest(*, rebuild: bool = False) -> dict[str, FAISS]:
    """扫描 `data/knowledge`：默认增量写入新 chunk 并软删旧切片。"""
    return ingest_indexes(persist=True, rebuild=rebuild)


if __name__ == "__main__":
    stores = ingest()
    total = sum(store.index.ntotal for store in stores.values())
    print(f"indexed {total} vectors across {len(stores)} tenants")
