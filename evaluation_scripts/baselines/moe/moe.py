# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import HumanMessage, SystemMessage

# # Create an instance of Ollama
# llm = Ollama(base_url="http://127.0.0.1:11434", model="mixtral:8x7b-instruct-v0.1-q5_K_M")


# def chat(tweets=None):
#     # Construct the prompt
#     # prompt = ChatPromptTemplate.from_messages([
#     #     ("system", "Given the following set of tweets (formatted in Tweet ID: Tweet), does the tweet have a positive or negative sentiment toward the topic on Enhanced Defense Cooperation Agreement (EDCA). For example, if the tweets mentions pro-Chinese military/political invasion then you should classify it as anti-EDCA (since EDCA is about US-Philippines alliance). Return your response in Json format only, don't contain other texts besides the json object."),
#     #     ("user", f"{tweets}")
#     # ])
#     # # Chain the prompt and llm
#     # chain = prompt | llm
#     # # Invoke the chain
#     # response = chain.invoke({})
#     # return response

    
#     prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")

#     # using LangChain Expressive Language chain syntax
#     # learn more about the LCEL on
#     # https://python.langchain.com/docs/expression_language/why
#     chain = prompt | llm | StrOutputParser()

#     # for brevity, response is printed in terminal
#     # You can use LangServe to deploy your application for
#     # production
#     print(chain.invoke({"topic": "Space travel"}))
#     return "Done"

# if __name__ == "__main__":
#     # Provide the tweets as input
#     tweets = "I hate Joe Biden"
#     # Call the chat function
#     result = chat(tweets)
#     # Print the result
#     print(result)


from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from few_shots import prompt as few_shots_example_prompt

# Create an instance of Ollama
llm = Ollama(base_url="http://127.0.0.1:11434", model="mixtral:8x7b-instruct-v0.1-q5_K_M")


def chat(tweets=None):
    # messages = [
    #     SystemMessage(content="You're a helpful assistant"),
    #     HumanMessage(content="What is the purpose of model regularization?"),
    # ]
    # # prompt = ChatPromptTemplate.from_template("Tell me a short joke about {topic}")
    # # chain = prompt | llm | StrOutputParser()

    # # print(chain.invoke({"topic": "Space travel"}))
    # print(llm.invoke(messages))
    # return "Done"

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a wondrous wizard of math."),
            few_shots_example_prompt,
            ("human", "{input}"),
        ]
    )
    chain = final_prompt | llm | StrOutputParser()
    response = chain.invoke({"input": "Given the following tweet format in Tweet ID: Tweet, does the tweet have a supportive (positive), negative (opposing), or neutral sentiment toward the topic: Tweet 3543190491230: RT @Richeydarian As I wrote back in 2019, the key Philippine bases as far as the Taiwan crisis is concerned are far to the north, namely in Fuga and Mavulis, which are NOT part of the EDCA bases, so far. https://t.co/TPukXHiC64"})

    print(response)

def main():
    chat()

if __name__ == "__main__":
    main()
