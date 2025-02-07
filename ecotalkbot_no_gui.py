from langchain_openai import ChatOpenAI
#import streamlit as st
import openai
import hmac
#from langchain.memory import StreamlitChatMessageHistory
#from langchain.chains import LLMChain
#from sentence_transformers import CrossEncoder
from langchain_core.messages.base import BaseMessage
#from sentence_transformers import SentenceTransformer
import json
import re
import datetime
import os
import json
import weaviate
import weaviate.classes.query as wq
from weaviate.classes.query import Filter
from FlagEmbedding import BGEM3FlagModel
from weaviate.classes.init import Auth
#from streamlit_float import *




wcd_url = os.environ["WCD_URL"]
wcd_api_key = os.environ["WCD_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(wcd_api_key),             # Replace with your Weaviate Cloud key
)
print(client.is_ready())
#st.write(client.is_ready()) 
#cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2') #sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
#cross_encoder = CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)
# @st.cache_resource
# def load_model():
#     return CrossEncoder('amberoad/bert-multilingual-passage-reranking-msmarco', max_length=512)

# cross_encoder = load_model()

chunks = client.collections.get("DocumentChunk")



output_dir = 'output_dir/'
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


openai.api_key = os.environ["OPENAI_API_KEY"]



    
#gpt3_5 = load_gpt3_5()
gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=apikey)



msgs = []



template = """
You are an expert in farmland biodiversity.

Your role is to assist a wide range of stakeholders in a Danish context, including:
* Danish farmers (organic and non-organic)
* Consultants for farmer organizations
* Municipal workers
* NGO representatives
* Professionals in food-related industries (associations, producers, retailers)
* Financial institutions (banks, pension funds)
* Interested citizens

Your primary tasks:
* Help farmers understand farmland biodiversity, identify practical ways to enhance it on their land, and solve challenges related to biodiversity practices.
* Guide other stakeholders in understanding farmland biodiversity, its relevance to their work or interests, and how it can be measured or applied meaningfully.

Your ultimate goal:
To provide actionable insights, foster understanding, and inspire practices that improve farmland biodiversity for sustainable, long-term benefits.

Use the pieces of retrieved information provided below to answer the user's question. 

Answer in English if the latest user query is in English, and in Danish if the latest user query is in Danish. Be helpful. Volunteer additional information where relevant, but keep it concise. 
Don't try to make up answers that are not supported by the retrieved information. If no suitable documents were found or the retrieved documents do not contain sufficient information to answer the question, say so.
Be critical of the information provided if needed. Mention the most impactful information first. Display formulas correctly, e.g. translating '\sum' to the sum symbol 'Σ'.
Try to keep the conversation going. For example, ask the user if they are interested in a related/neighboring topic, or would like more detail on something. For example if they are interested in the lapwing, they may also be interested in other relevant birds, such as the skylark.

Include references in your answer to the documents you used, to indicate where the information comes from. The documents are numbered. Use those numbers to refer to them. Use the term 'Document' followed by the number, e.g. '(Document 1)' or '(Document 2, Document 5)' when citing multiple documents. Do not cite other sources than the provided documents. Do not list the sources below your answer. They will be provided by a different component.

Retrieved information:
{context}

Preceeding conversation:
{conversation}

Question: {question}
Helpful Answer:"""

contextualizing_template = """ Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. The overall topic is biodiversity.
Do NOT answer the question, just reformulate it if needed and otherwise return it as is. Keep it in the original language.

Chat history:
{history}

Latest user question:
{question}

Standalone version of the question:
"""


meta_fields = ["title", "section_headers", "link", "year", "target_audience", "geography", "keywords", "abstract", "type_of_information"]
def format_docs(docs):
    if docs == []:
        return "Ingen relevante kilder blev fundet."#"No relevant documents were found."
    doclist = []
    for d in docs:
        nd = {"page_content":d.properties["page_content"], "metadata":{}}
        for field in meta_fields:
            nd["metadata"][field] = d.properties.get(field)
        doclist.append(nd)
    return"\n\n".join( str(num+1)+') '+doc["page_content"]+'\n'+json.dumps(doc["metadata"], indent=4) for num, doc in enumerate(doclist))



#question = st.text_input("Write a question about Gaia: ", key="input")
    
    

############################################
#Refence handling

def add_sources(docs, source_numbers):
    lines = []
    #lines.append('\nSources:')
    if docs and source_numbers:
        print(source_numbers)
        for count, num in enumerate(source_numbers):
            rd = docs[int(num)-1]
            doc_info = []
            title = rd.properties["title"].replace('.', '\.')
            doc_info.append(str(count+1)+'. '+title)
            section_info = rd.properties["section_headers"]
            if section_info:
                doc_info.append('  \n   (Afsnit: '+', '.join(section_info)+')')
            else:
                doc_info.append('  \n   (Afsnti: '+rd.properties["page_content"][:50]+'...)')
            doc_info.append('  \n'+rd.properties["link"])
            lines.append(''.join(doc_info))
    #text = '\"\"\"'+'\n'.join(lines)+'\"\"\"'
    else:
        lines = ["De oplysninger, der præsenteres her, refererer ikke eksplicit til de kilder, der blev udvalgt til EcoTalkBot-projektet. Der kan være behov for ekstra forsigtighed med hensyn til nøjagtighed."]
        #lines = ["The information presented here does not explicitly reference the sources that were selected for the EcoTalkBot project. Extra caution with respect to accuracy may be in order."]
    return '  \n'.join(lines)

def replace_in_text(x, y, text):
    # Define the regex pattern to match 'Document x' with exact match on x
    pattern = rf'Document {x}(?=\b|\D)'
    # Define the replacement text 'Document y'
    replacement = f'Kilde {y}'
    # Use re.sub() to replace all instances of 'Document x' with 'Document y'
    updated_text = re.sub(pattern, replacement, text) 
    return updated_text

def replace_documents_list(text):
    # Define the regex pattern to match '(Documents x, y, z)'
    pattern = r'\(Documents? (\d+(?:, \d+)*(?:,? and \d+)?)\)'
    # Replacement function to reformat the matched text
    def replacement_function(match):
        # Extract the list of numbers from the match
        numbers = match.group(1)
        #print(numbers)
        number_list = re.split(r', and |, | and ', numbers)
        #print(number_list)
        # Join each number with 'Document ' prefix
        new_text = ', '.join([f'Document {num}' for num in number_list])
        #print(new_text)
        # Return the formatted text in the desired format
        return f'({new_text})'
    # Use re.sub() to replace all instances of '(Documents x, y, z)' with '(Document x, Document y, Document z)'
    updated_text = re.sub(pattern, replacement_function, text)
    return updated_text

def f7(seq): #deduplication of list while keeping order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]
    
def used_sources(answer, lendocs):
    listed_pattern = r'\d+, ?\d'
    listed = re.findall(listed_pattern, answer)
    if listed:
        answer = replace_documents_list(answer)
    pattern = r'Document \d+'
    used = re.findall(pattern, answer)
    used = f7(used)
    used = [u.split()[-1] for u in used]
    remove = [u for u in used if int(u) > lendocs]
    used = [u for u in used if not u in remove]
    for num, u in enumerate(used):
        answer = replace_in_text(u, str(num+1), answer)
    if remove:
        for rn in remove:
            if '(Document '+rn+')' in answer:
                answer = answer.replace('(Document '+rn+')', '')
            elif 'Document '+rn+',' in answer:
                answer = answer.replace('Document '+rn+', ', '')
            elif ', Document '+rn in answer:
                answer = answer.replace(', Document '+rn, '')
    return answer, used
#############################################################################
model = BGEM3FlagModel('BAAI/bge-m3')

def vectorize(query):
    sentences = [query]
    embeddings = model.encode(sentences, batch_size=5, max_length=1024, return_dense=True, return_sparse=True)
    return embeddings['dense_vecs'][0]

store_fields = ["parent_doc", "chunk_number", "title", "section_headers", "page_content", "link", "year", "target_audience", "geography", "keywords", "data_type", "type_of_information"]

#with open("used_sources.json", "r") as f:
#    source_data = json.load(f)
#########################################################



#new_msg = BaseMessage(type='ai', content="Velkommen til EcoTalkBot – Tal med mig om biodiversitet på landbrugsjord  \n\n  Hvordan kan jeg hjælpe dig?")
msgs.append(("ai", "Velkommen til EcoTalkBot – Tal med mig om biodiversitet på landbrugsjord  \n  Hvordan kan jeg hjælpe dig?"))
print("ai: Velkommen til EcoTalkBot – Tal med mig om biodiversitet på landbrugsjord  \n  Hvordan kan jeg hjælpe dig?")

active = True

while active:
    user_input = input("human: ")
    if user_input.strip() == "stop":
        active = False
        client.close()
    else:
        prev_conv = '\n'.join([msgtype+': '+msgcontent for msgtype, msgcontent in msgs[-4:]])
        user_msg = ("human", user_input)
        msgs.append(user_msg)
        time = datetime.datetime.now()
        filename = str(time)+'.json'
        path = output_dir+filename
        interaction = {"date_time":str(time)}
        interaction["user_input"] = user_input
        contextualizing_prompt = contextualizing_template.format(history=prev_conv, question=user_input)
        #print(contextualizing_prompt)
        contextualized_result = gpt4.invoke(contextualizing_prompt)
        search_query = contextualized_result.content
        print("contextualized_prompt: "+search_query)
        interaction["contextualized_query"] = search_query
        interaction["previous_interactions"] = prev_conv
        query_vector = vectorize(search_query)
        response = chunks.query.near_vector(
            #filters=Filter.by_property("target_audience").contains_any(['farmer', 'all', 'consultant']),
            near_vector=query_vector,  # A list of floating point numbers
            limit=7,
            return_metadata=wq.MetadataQuery(distance=True),
            )
        docs = response.objects
        # if len(docs) < 7:
        #     no_filter = chunks.query.near_vector(
        #         near_vector=query_vector,  # A list of floating point numbers
        #         limit=7-len(docs),
        #         return_metadata=wq.MetadataQuery(distance=True),
        #         )
        #     docs.extend(no_filter.objects)
        interaction["retrieved_documents"] = []
        for d in docs:
            docjson = {}
            for pf in store_fields:
                docjson[pf] = d.properties[pf]
                docjson["distance_to_query"] = d.metadata.distance
            interaction["retrieved_documents"].append(docjson) 
        try:
            full_prompt = template.format(context=format_docs(docs), question=user_input, conversation=prev_conv)
            result = gpt4.invoke(full_prompt)
            ai_answer, source_numbers = used_sources(result.content, len(docs))
            sources = add_sources(docs, source_numbers)
        except ValueError:
            ai_answer = ''
            with open(path, 'w') as f:
                json.dump(interaction, f)
        if not ai_answer:
            print('Ups, der gik noget galt. Prøv venligts igen.')
        else:
            print("ai: "+ai_answer)
            print()
            print("sources:")
            print(sources) 
            print("\n")
            #ai_msg = BaseMessage(type="ai", content=ai_answer)
            #setattr(ai_msg, 'sources', sources)
            msgs.append(("ai", ai_answer))    
            #st.session_state.content = ''
            interaction["original_answer"] = result.content
            interaction["sources"] = sources
            interaction["final_answer"] = ai_answer
            #filename = str(time)+'.json'
            #path = output_dir+filename
            with open(path, 'w') as f:
                json.dump(interaction, f)



    
   

###########################################################

# tempyear = '''Title: **{title}**  
# Year: {year}  
# Author: {author}  
# Link: {link}
# '''

# temp = '''Title: **{title}**  
# Author: {author}  
# Link: {link}
# '''

# with tab2:
#     st.header("Liste af alle kilder som chatbotten kan søge i:")
#     for d in source_data:
#         #if source_data[d]["extended (Dec 16 2024)"] == "x":
#         source_data[d]["Title"] = source_data[d]["Title"].strip().replace('.', '\.')
#         #st.markdown("| "+" | ".join([str(source_data[d].get("Title")), str(source_data[d].get("Year")), str(source_data[d].get("Author")), source_data[d]["Link"]])+" |")
#         if source_data[d]["Year"] == 'x' or source_data[d]["Year"] == None:
#             st.markdown(temp.format(title=str(source_data[d].get("Title")), author=str(source_data[d].get("Author")), link=source_data[d]["Link"]))
#         else:
#             st.markdown(tempyear.format(title=str(source_data[d].get("Title")), year=str(source_data[d].get("Year")), author=str(source_data[d].get("Author")), link=source_data[d]["Link"]))
   
    
    
# Check the result of the query

# Check the source document from where we 
# for rd in result["source_documents"]:
#     print(rd)
# print('\n')
# print(result["result"])