# coding:utf-8 
# from langchain.docstore.document import Document


def show_result(result, call_type="text"):
    query = result["query"]
    answer = result["result"]
    source_documents = result["source_documents"]
    dict_src_doc = []
    for x in source_documents:
        # {'source': '/home/waf_guide.md', 'page_number': 1, 'category': 'UncategorizedText'}
        dict_src_doc.append(dict(page_content=x.page_content, metadata=x.metadata,) )
    if call_type == "text":
        print(f"""
问题：{query} 
回答：{answer}

【引用】""")
        for x in dict_src_doc:
            # print(f"\t\t {x['metadata']['source']} ({x['metadata']['category']}) -- {x['page_content']}")
            print(x)
        print("----------------------")
    return dict(refs=dict_src_doc, query=query, answer=answer)
