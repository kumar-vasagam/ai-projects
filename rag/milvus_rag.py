from pymilvus import MilvusClient
from pymilvus import model
import random
import numpy as np
from langchain_milvus import Milvus
from dotenv import load_dotenv
import os, json
import pypdf
from langchain_core.documents import Document
from openai import OpenAI
from langchain.agents import create_agent
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

class MilvusRag():
    client = MilvusClient("milvus_demo.db")
    load_dotenv()
    OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
    embedding_model="nvidia/nemotron-3-embed-1b:free"

def hello_milvus():
    mg = MilvusRag()
    if mg.client.has_collection(collection_name="demo_collection"):
        mg.client.drop_collection(collection_name="demo_collection")

    mg.client.create_collection(collection_name="demo_collection", dimension=2048)

    docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",]

    # Use fake representation with random vectors (768 dimension).
    vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
    data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
    res = mg.client.insert(
    collection_name="demo_collection",
    data=data)

    res = mg.client.search(
    collection_name="demo_collection",
    data=[vectors[0]],
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],)
    print(res)

    res = mg.client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],)
    print(res)

def embed_with_openrouter(input: str) -> str:
    mg = MilvusRag()
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=mg.OPENROUTER_API_KEY)
    cer = client.embeddings.create(input=input, model=mg.embedding_model)
    
    return cer.data[0].embedding

def load_meta_data(areas: list[dict]):
    meta_collection = "metadata"
    mg = MilvusRag()
    if mg.client.has_collection(collection_name=meta_collection):
        mg.client.drop_collection(collection_name=meta_collection)
    mg.client.create_collection(
                    collection_name=meta_collection,
                    dimension=2048
                )
    for idx, item in enumerate(areas):
        t = item['topic']
        f = item['file_path']
        print(f"topic {t} and file_path {f}")
        with open(f) as curr_file:
            content = curr_file.read()
            mg.client.insert(collection_name=meta_collection, 
                             data={"id" : idx, "topic":t, "vector":embed_with_openrouter(content), "content":content})
            print(f"meta data loaded for {t} ok!")
    print("layer1 loaded ok!")
    
    

def load_from_pdf(collection_name: str, file_path: str):
    mg = MilvusRag()
    if mg.client.has_collection(collection_name=collection_name):
        return
        #mg.client.drop_collection(collection_name=collection_name)

    mg.client.create_collection(
            collection_name=collection_name,
            dimension=2048
        )
    reader = pypdf.PdfReader(file_path)
    data = []
    for i, page in enumerate(reader.pages):
        raw_text = page.extract_text()
        data.append({"id":i, "vector": embed_with_openrouter(raw_text), "text":raw_text})
    
    out = mg.client.insert(collection_name=collection_name, data=data)
    print(out)



def get_qa_pairs() -> dict[str, str]:
    return {
        "What are the minimum dwelling unit and use requirements for a Property to be eligible under Freddie Mac guidelines?":
            "The Property must contain five or more dwelling units, must be designed (in whole or in part) for residential use, each residential unit must contain kitchen and bathroom facilities, and it must be served by public water and sanitary sewer systems.",
        "What restrictions are placed on the Borrower regarding home sharing platforms like Airbnb or VRBO?":
            "The Borrower must not participate in home sharing activities (short-term rentals typically under one month marketed via peer-to-peer marketplaces) or enter into leases (including master leases) that the Borrower knows or should have known are intended for full- or part-time home sharing.",
        "When is a wood-damaging insect inspection report not required?":
            "When the Property has no wood framing or structural members (significant components subject to damage by wood-damaging insects) as determined by either the Property Condition Report or the Physical Risk Report; or if the Borrower has a contract in place that remains for the mortgage term and there is no evidence of wood damage.",
        "What are the minimum routine inspection intervals required under a Moisture Management Plan (MMP)?":
            "At a minimum, inspections must occur annually for all common areas and areas with a past history of water intrusion, leaks, or mold, and at unit turnover or at a tenant's request for all units.",
        "What is Freddie Mac's standard occupancy requirement prior to loan closing and delivery?":
            "For the three consecutive months prior to loan closing and as of the Delivery Date, at least 90% of the living units (or higher if needed to cover debt service and expenses) must have been occupied at rent levels supporting the Underwriting Value.",
        "If a Property is legal non-conforming per the zoning analysis in an appraisal, what does Freddie Mac require?":
            "Ordinance and Law Insurance per Section 31.12 and a non-conforming carveout are required.",
        'What four conditions define an "Independent Property"?':
            "(1) Direct access to a publicly dedicated and maintained street without reliance on a Shared Access Agreement; (2) contains needed Essential Facilities; (3) contains needed Recreational Facilities; and (4) is financially viable and independent of all other properties.",
        "What are the tax parcel requirements for a mortgaged Property?":
            "The Property must be identified as a single tax parcel or constitute the entirety of multiple tax parcels, and cannot include property not subject to the Mortgage sold to Freddie Mac.",
        "Can a Shared Access Agreement permit a loss of use in the event of a breach by a party?":
            "No, the agreement may not allow for loss of use in the event of a breach, though it may permit the placement of a subordinate lien for unpaid maintenance costs.",
        "Which commercial leases require an executed tenant estoppel?":
            "(1) Individual commercial leases accounting for 5% or more of gross potential rent; (2) all commercial leases over 1,000 sq. ft. if total commercial lease income is 10% or more of gross potential rent; and (3) any lease specifically requested by Freddie Mac.",
        "What are the subordination rules for a commercial lease between the Borrower and an Affiliate?":
            "It must be subordinate to the Mortgage lien unless it can be terminated by the Property owner with or without cause on 30 days' notice without fee or penalty; Freddie Mac will not agree to a nondisturbance agreement for an Affiliate lease.",
        "What maximum LTV and minimum DCR apply if Freddie Mac consents to Subordinate Financing?":
            "Combined debt may not exceed an 85% LTV Ratio, and combined debt service may not result in a DCR below 1.20x.",
        "When is a third-party property management company mandatory?":
            "When all four are met: (1) UPB ≤ $10M; (2) transaction includes a First-Time, Rapid Growth, or Limited Experience Sponsor; (3) no controlling individual/entity lives or has an office within 100 miles; and (4) no controlling party has owned ≥5 properties for 5 years in the market.",
        "Name two conditions that make a Mortgage entirely ineligible for purchase by Freddie Mac.":
            "Any of the following: PML > 40% in an Elevated Seismic Hazard Region without retrofit; encumbered by a Private Transfer Fee Covenant created on/after Feb 8, 2011; located in an SFHA where the community does not participate in NFIP or flood insurance is lacking; or encumbered by a cross-regulatory agreement encumbering other property.",
        "May a Seller hire a third-party contractor to conduct the property inspection?":
            "No, an inspector familiar with evaluating multifamily asset quality must perform it, and a third-party contractor may not perform the inspection.",
        "What percentage of units must be inspected during full underwriting for a Property with more than 30 units?":
            "10% of units, with no fewer than 10 units and no more than 30 units (excluding Down Units and commercial units, both of which must be 100% inspected), with at least 50% being occupied units.",
        "What is the shelf life/validity period for a completed property inspection?":
            "The inspection must have been completed within 90 days of Freddie Mac's receipt of the applicable underwriting package; otherwise, a new inspection must be performed (no recertification allowed).",
        "What are the mandatory completion timeframes for PR-90 Priority Repairs versus all other Priority Repairs?":
            "PR-90 Repairs must be completed within 90 days after the Origination Date; all other Priority Repairs must be addressed as soon as possible and completed within 365 days after the Origination Date.",
        "What control requirements apply to the Borrower in a Fractured or Partial Condominium?":
            "The Borrower must own a majority of Condominium Units, control a majority of the board of directors, and hold sufficient voting rights to control specified governance and budgetary matters.",
        "What happens if a loan is underwritten with abated taxes, but the Borrower fails to obtain the tax abatement within 12 months?":
            "Freddie Mac may require the Borrower to partially prepay the Mortgage (calculated as the difference supported with the abatement vs. full taxes) along with any applicable prepayment premium.",
        "Is financing permitted for a Borrower-owned Solar Electric System?":
            "No, no financing of a Borrower-owned Solar Electric System is permitted.",
        "Will Freddie Mac approve an Infrastructure Agreement that assigns the Borrower's reversionary or lessor interest?":
            'No, Freddie Mac will not permit or approve an Infrastructure Agreement (including a Solar Agreement) that purports to assign the Borrower\'s interest as "lessor" (or reversionary interest) to a third party.',
    }

def hypothetical_question_pattern():
    # pre-load
    load_hypotheticals()
    # query_regular_collection("what are recreational facilities?", "mf_property")
    q1 = "what are recreational facilities?"
    q2 = "what conditions define an independent property?"
    query_real_data(q2, "hypotheticals")


def query_real_data(query: str, collection_name:str) -> str:
    """
    This function queries the relevant collection and returns the data pertaining to the question.
    Args:
        query (str): The question to be asked.
        collection_name (str): The name of the collection to query for relevant data. The valid values are 
        {topic} key retrieved from `query_meta_data()` tool. 
    Returns:
        str: A JSON string containing the retrieved lines and their distances from the query.
    """
    mg = MilvusRag()
    mg.client.load_collection(
        collection_name=collection_name
    )
    print("What is the collecitn name coming in?", collection_name)
    out = mg.client.search(collection_name=collection_name, 
                     data=[embed_with_openrouter(query)],
                     output_fields=["text"], limit=4)
    print(out[0])
    retrieved_lines_with_distances = [(res["entity"]["text"], res["distance"]) for res in out[0]]
    return json.dumps(retrieved_lines_with_distances, indent=4)

def load_hypotheticals():
    hypo_collection = "hypotheticals"
    mg = MilvusRag()
    if mg.client.has_collection(collection_name=hypo_collection):
        mg.client.drop_collection(collection_name=hypo_collection)

    data = []

    for i, (k, v) in enumerate(get_qa_pairs().items()):
        data.append({"id":i, "vector": embed_with_openrouter(k), "text":k})

    mg = MilvusRag()
        
    mg.client.create_collection(
                collection_name=hypo_collection,
                dimension=2048
            )    
    out = mg.client.insert(collection_name=hypo_collection, data=data)
    print(">>> Hypos loaded ok!")

def query_meta_data(question: str, collection_name="metadata") -> dict:
    """
    This meta data function provides which topic that this question belongs to. It might return multiple
    topics if there is no clear answer. The caller is expected to pick the best topic that matches the question.
    The caller is expected to use the topic as the collection_name for the next query to get the answer to the question.
    Args:
        question (str): User's question.
        collection_name (str): The name of the collection to query for metadata. Default: 'metatdata'
    Returns: 
        Dict of {topic: <actual_topic_name>, content: <actual_content>}. If the caller finds the content to be aligned to the question, then they should 
        further query the collection {actual_topic_name} in the next step.
    """
    mg = MilvusRag()
    mg.client.load_collection(
        collection_name=collection_name
        )
    out = mg.client.search(collection_name=collection_name, 
                     data=[embed_with_openrouter(question)],
                     output_fields=["topic", "content"])
    top_content = {}
    print("lgt of out -- ", len(out))
    for i, value in enumerate(out):
        print(f"Result {i}:")
        for res in value:
            print(f"Topic: {res['entity']['topic']}, Content: {res['entity']['content']}, Distance: {res['distance']}")
            top_content[res['entity']['topic']] = res['entity']['content']
    return top_content

def get_final_response(question: str, chunks: str) -> str:
    """
    This function provides final response to the user's question. It is expected that the caller will fill in the chunks 
    with the relevant information retrieved from the DB along with the user's original question
    Args:
        question(str): The user's original question
        chunks (str): The relevant information retrieved from the DB.
    """
    llm_system_prompt = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
       Use the following source documents to answer the user's questions.
       If you don't know the answer, just say that you don't know.
       Use three sentences maximum and keep the answer concise.
        <context>
            {chunks}
        </context>
       """
    inferencing_model = "nvidia/nemotron-3.5-lightning:free" # gpt-5.4-mini
    mg = MilvusRag()
    client = OpenAI(api_key=mg.OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    response = client.chat.completions.create(model=inferencing_model, messages=[
        {"role": "system", "content": llm_system_prompt},
        {"role": "user", "content": question},
    ])
    res = response.choices[0].message.content
    print(">>> ", res)
    return res
    # llm = ChatOpenAI(model=inferencing_model, temperature=1)
    # ai_msg = llm.invoke([
    #         {"role": "system", "content": llm_system_prompt},
    #         {"role": "user", "content": question},
    #     ],
    # )
    # print(">>>", ai_msg.content)
    # return ai_msg.content['answer']
    # return {"answer": ai_msg.content, "documents": question}
    

def ask_a_question(question: str):
    """
    Given a question you should first query the meta data collection and get the topic of interest
    Then with the topic, get the information chunk from the DB that answers the relevant question. The topic key returned in Step 1, verbatim, is the collection name for this step
    Finally get the well formulated response to the user's question. 
    Use the tools that are provided to you. Do not use any other tool.
    """
    # query_meta_data(question=question, collection_name="layer1")
    agent_system_prompt = "You are a helpful assistant. " \
    "Given a question you should first query the meta data collection and get the topic of interest" \
    "Then with the topic, get the information chunk from the DB that answers the relevant question. The topic key returned in Step 1, verbatim, is the collection name for this step" \
    "Finally get the well formulated response to the user's question." \
    "The detailed information are stored in discrete collections, so follow these steps in EXACT order" \
    "Use the tools that are provided to you. Do not use any other tool"
    
    da = create_deep_agent(model="openrouter:deepseek/deepseek-v4-flash", system_prompt=agent_system_prompt, 
                           tools=[query_meta_data, query_real_data, get_final_response])
    out = da.invoke({"messages": [{"role": "user", "content": question}]})
    print("answer - ", out)
    

def main():
    # hello_milvus()
    # hypothetical_question_pattern()
    layer1 = [
        {"topic": "servicing", "file_path": "./servicing.txt"},
        {"topic": "property", "file_path": "./property.txt"}
    ]
    load_meta_data(layer1)

    layer2 = [
            {"topic": "servicing", "file_path": "/Users/kumaresan/Documents/mf_guide_ch_36.pdf"},
            {"topic": "property", "file_path": "/Users/kumaresan/Documents/mf_guide_ch_8.pdf"}
        ]
    for layer in layer2:
        load_from_pdf(layer['topic'], layer['file_path'])
    question = "Is financing permitted for a Borrower-owned Solar Electric System?"
    ask_a_question(question)


if __name__ == "__main__":
    main()