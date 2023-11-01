# region Imports
import os
import sys
from dotenv import main
from langchain.indexes import GraphIndexCreator
from langchain.chains import GraphQAChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader 
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import matplotlib.pyplot as plt
import networkx as nx
# endregion

# region Data
transcript_name_directory = r'.\transcripts'
full_text_dump = ''

for filename in os.listdir(transcript_name_directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(transcript_name_directory, filename)
        with open(file_path, 'r') as file:
            content = file.read()
            full_text_dump += content + " "
            # print(f'File: {filename}\nContent:\n{content}\n')
# endregion

# region GraphIndexCreator
index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))

graph = index_creator.from_text(full_text_dump)
print(graph.get_triples())


# Create graph
G = nx.DiGraph()
G.add_edges_from((source, target, {'relation': relation}) for source, relation, target in graph.get_triples())

# Plot the graph
plt.figure(figsize=(8,5), dpi=300)
pos = nx.spring_layout(G, k=3, seed=0)

nx.draw_networkx_nodes(G, pos, node_size=1000)
nx.draw_networkx_edges(G, pos, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=6)
edge_labels = nx.get_edge_attributes(G, 'relation')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5)

# Display the plot
plt.axis('off')
plt.show()
# endregion

# region VectorstoreIndexCreator
main.load_dotenv()
os.environ["OPENAI_API_KEY"] = str(os.getenv("OPENAI_API_KEY"))

loader = DirectoryLoader(transcript_name_directory, glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
print(f'{index}')
# endregion

# region VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0)

qa_chain = load_qa_chain(llm, chain_type="map_reduce")

qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)

result = qa_document_chain.run(
    input_document=full_text_dump, 
    question="who is Jack"
)
print(f'{result}')
# endregion