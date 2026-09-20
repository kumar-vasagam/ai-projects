from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import FewShotChatMessagePromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# MessageType Use one of 'human', 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'.
def connect_with_llm() -> ChatHuggingFace:
    llm = HuggingFaceEndpoint(
        repo_id="microsoft/Phi-3-mini-4k-instruct",
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        repetition_penalty=1.03,
        temperature=1.0
        )
    chat = ChatHuggingFace(llm=llm, verbose=True)
    return chat

def zero_shot_learning():
    chat = connect_with_llm()
    messages = [("system", "you are a classifier of communication stlyes in one word. Acceptable classifications are Professional, Immature, Casual"), ("human", "classify this text : Thank you for your email, will get back to you by tomorrow")]
    response = chat.invoke(messages)
    print("response metadata -> ", response.response_metadata)
    print("zero shot classification output : ", response.content)

def one_shot_learning():
    chat = connect_with_llm()
    # messages = [("system", "you are a classifier of communication stlyes in one word. Acceptable classifications are Professional, Immature, Casual"), ("human", "classify this text : Thank you for your email, will get back to you by tomorrow")]
    messages = [("system", "You are a text classifier. Classify the text"), 
                ("user", "Thanks for your email, will reply tomorrow"), 
                ("assistant", "Professional"),
                ("user", "k, thx! will get back to ya!")]
    response = chat.invoke(messages)
    print("response metadata -> ", response.response_metadata)
    print("one shot classification output : ", response.content)

def multi_shot_learning(input_messages : list[tuple[str, str]]):
    chat = connect_with_llm()
    # messages = [("system", "you are a classifier of communication stlyes in one word. Acceptable classifications are Professional, Immature, Casual"), ("human", "classify this text : Thank you for your email, will get back to you by tomorrow")]
    messages = [("system", "You are a sentiment analyst. Classify the sentiment in one word"), 
                ("user", "i love pancake!"),("assistant", "Positive"),
                ("user", "I don't like this movie"), ("assistant", "Negative"),
                ("user", "The drink was fine"), ("assistant", "Neutral")]
    output = []
    for m in input_messages:
        messages.append(m)
        response = chat.invoke(messages)
        output.append(response.content)
        print("sentiment of ' ", m[1], "' is ", response.content)
        messages.pop(len(messages)-1) # resetting the base prompt
    
    print(output)
    

def multi_shot_prompter():
    messages = [("user", "this cell phone sucks"), ("user", "this code rocks"), ("user", "the traffic was okay")]
    multi_shot_learning(messages)
    

def few_shot_learning():
    # example_prompt = PromptTemplate.from_template("Question: {question}\n{answer}")
    # examples = [{"question" : "Email: 'Thank you for your email, will get back to you by tomorrow'", "answer" : "formal"},
    #             {"question" : "Email: 'Sure thing!, I am on it :) '", "answer" : "casual"}]
    
    # print(example_prompt.invoke(examples[0]).to_string())

    # prompt = FewShotPromptTemplate(example_prompt = example_prompt, examples=examples, suffix="Email: {input}", input_variables=["input"],)
    # pv = prompt.invoke({"input": "k, thx, will get back to ya!"})
    # print(pv.to_string())
    examples = [{"input":'Thank you for your email, will get back to you by tomorrow', "output" : "formal"},
                {"input":'Sure thing!, I am on it :)', "output" : "casual"}]

    messages = [{"role" : "system", "content" : "You are a helpful assistant"},
                {"role" : "user", "content" : "classify this text - Thank you for your email, will get back to you by tomorrow"},
                {"role" : "assistant", "content" : "Professional"},
                {"role" : "user", "content" : "classify this text - Sure thing!, I am on it :)"},
                {"role" : "assistant", "content" : "Casual"},
                {"role" : "user", "content" : "classify this text - k, thx! will get back to ya!"}]
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("user", "{content}"),
            ("assistant", "{content}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt, examples=examples)
    print(few_shot_prompt.invoke({}).to_messages())

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        few_shot_prompt,
        ("user", "{input}"),
    ])

    chat = connect_with_llm()
    chain = final_prompt | chat
    response = chain.invoke({"input" : "k, thanks, will get back to ya!"})
    # response = chat.invoke("k, thx, will get back to ya!")
    # response.to_json
    print("few shot output - ", response.to_json())


if __name__ == '__main__':
    zero_shot_learning()
    one_shot_learning()
    print("----------")
    multi_shot_prompter()
    # few_shot_learning()
