from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
few_shots_examples = [
    {
        "question": "Given the following tweet format in Tweet ID: Tweet, does the tweet have a supportive (positive), negative (opposing), or neutral sentiment toward the topic: Tweet 1412874109381: RT @GordianKnotRay “The 🇺🇸United States stands with The 🇵🇭#Philippines in the face of the People’s Republic of 🇨🇳#China Coast Guard’s continued infringement upon freedom of navigation in the #SouthChinaSea…We call upon Beijing to desist from its provocative and unsafe conduct.” — @StateDeptSpox https://t.co/3I8n829W2d",
        "answer": """
                    This tweet has positive (pro) sentiment on EDCA because it is againt Chinese military and political invasion in Philippines, which aligns with the purpose of EDCA (Enhanced Defense Cooperation Agreement) that supports US-Philippines alliance. So the output is:
                    {
                        ID: 1412874109381,
                        Sentiment: supportive
                    }
                    """,
    },
    {
        "question": "Given the following tweet format in Tweet ID: Tweet, does the tweet have a supportive (positive), negative (opposing), or neutral sentiment toward the topic: Tweet 141239233123: RT @Richeydarian This is BIG! Just weeks after MARCOS Jr. visit to CHINA, which produced no clear breakthrough, the philippines is not only implememting EDCA, but has now likely agreed to US access to key bases near TAIWAN! Huge geopolitical implications…",
        "answer": """
                    This tweet has positive (pro) sentiment on EDCA. So the output is:
                    {
                        ID: 141239233123,
                        Sentiment: supportive
                    }
                """,
    },
    {
        "question": "Given the following tweet format in Tweet ID: Tweet, does the tweet have a supportive (positive), negative (opposing), or neutral sentiment toward the topic: Tweet 5438491410209: RT @KrstnnBonVoyage Not blaming any Presidents. But the idea of EDCA in the Philippines shouldn’t have happened to begin with. Why? It doesn’t take a genius to understand this. ",
        "answer": """
                    This tweet has negative (anti) sentiment on EDCA. So the output is:
                    {
                        ID: 5438491410209,
                        Sentiment: opposing
                    }
                """,
    },
    {
        "question": "Given the following tweet format in Tweet ID: Tweet, does the tweet have a supportive (positive), negative (opposing), or neutral sentiment toward the topic: Tweet 3543190491230: RT @Richeydarian As I wrote back in 2019, the key Philippine bases as far as the Taiwan crisis is concerned are far to the north, namely in Fuga and Mavulis, which are NOT part of the EDCA bases, so far. https://t.co/TPukXHiC64",
        "answer": """
                This tweet has negative (anti) sentiment on EDCA. So the output is:
                {
                    ID: 3543190491230,
                    Sentiment: supportive
                }
            """,
    },
    {
        "question": "Given the following tweet format in Tweet ID: Tweet, does the tweet have a supportive (positive), negative (opposing), or neutral sentiment toward the topic: Tweet 3543190492230: RT @Kanthan2030 In the Philippines, the President’s sister - who is also a Senator - warns against getting trapped in the U.S.-China rivalry. The US wants to turn Filipinos into Asia’s Ukrainians — sacrificial pawns in a future war against Beijing. https://t.co/DKkgivb6Iv",
        "answer": """
                This tweet has negative (anti) sentiment on EDCA. So the output is:
                {
                    ID: 3543190492230,
                    Sentiment: opposing
                }
            """,
    },
]

# few_shots_examples_prompt = PromptTemplate(
#     input_variables=["question", "answer"], template="Question: {question}\n{answer}"
# )

few_shots_examples_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
        ("ai", "{answer}"),
    ]
)

# ChatPromptTemplate

prompt = FewShotChatMessagePromptTemplate(
    examples=few_shots_examples,
    example_prompt=few_shots_examples_prompt,
    # suffix="Question: {input}",
    # input_variables=["input"],
)

# print(prompt.format())


