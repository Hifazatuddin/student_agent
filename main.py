import os
from dotenv import load_dotenv
from agents import Agent,Runner,AsyncOpenAI,OpenAIChatCompletionsModel

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

model=OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

English_agent = Agent(
    name="English",
    instructions="""You are assistant that helps user to answer 
    questions about English language and grammar.""",
    model=model,
)
Urduagent = Agent(
    name="Urdu",
    instructions="""You are assistant that helps user to answer 
    questions about Urdu language and grammar.""",
    model=model,

)

islamic_agent = Agent(
    name="Islamic studies",
    instructions="""You are assistant that helps user to answer 
    questions about Islamic studies, Quran, Hadith, and other related topics.""",
    model=model,

)
biology_agent = Agent(
    name="Biology",
    instructions="""You are assistant that helps user to answer 
    questions about Biology, including human anatomy, physiology, and other related topics.""",
    model=model,
    
)
mathematics_agent = Agent(
    name="Mathematics",
    instructions="""You are assistant that helps user to answer 
    questions about Mathematics, including algebra, geometry, calculus, and other related topics.""",
    model=model,

)
chemistry_agent = Agent(
    name="Chemistry",
    instructions="""You are assistant that helps user to answer 
    questions about Chemistry, including organic chemistry, inorganic chemistry, and other related topics.""",
    model=model,
)
physics_agent = Agent(
    name="Physics",
    instructions="""You are assistant that helps user to answer 
    questions about Physics, including mechanics, thermodynamics, and other related topics.""",
    model=model,
)
general_knowledge_agent = Agent(
    name="General Knowledge",
    instructions="""You are assistant that helps user to answer 
    questions about General Knowledge, including history, geography, and other related topics.""",
    model=model,
)
main_agent = Agent(
    name="main_agent",
    instructions="""You are a main agent that helps user to answer questions by delegating them to other agents.
    You will receive a question from the user and you will delegate it to the appropriate agent based on the topic""",
    handoffs=[English_agent, Urduagent, islamic_agent, biology_agent, mathematics_agent, chemistry_agent, physics_agent, general_knowledge_agent],
    model=model,
)

# i add user question here
user_question=input("please enter Your Question ?")

result=Runner.run_sync(
    main_agent,
    user_question
    )
print(result.final_output)