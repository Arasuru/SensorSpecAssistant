def preview_chunks(chunks):
    #preview 3 chunks
    for i, chunk in enumerate(chunks[:3]):
        print(f"--- Chunk {i+1} ---")
        print(f"chunk metadata: {chunk.metadata}")
        print(f"chunk content: {chunk.page_content}")
        print()