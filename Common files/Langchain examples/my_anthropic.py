from langchain_anthropic import ChatAnthropic

chatmodel = ChatAnthropic(api_key = "",model = "claude-3-sonnet-20240229", temperature=0.2, max_tokens=1000)

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("What is a good name for a company that makes {product}?")
prompt.format(product="colorful socks")

from langchain_core.prompts.chat import ChatPromptTemplate

template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chat_prompt.format_messages(input_language="English", output_language="French", text="I love programming.")

template = "Generate a list of 5 {text}.\n\n{format_instructions}"

from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
output_parser.parse("hi, bye")
# >> ['hi', 'bye']

chat_prompt = ChatPromptTemplate.from_template(template)
chat_prompt = chat_prompt.partial(format_instructions=output_parser.get_format_instructions())
chain = chat_prompt | chatmodel | output_parser
print(chain.invoke({"text": "colors"}))
# >> ['red', 'blue', 'green', 'yellow', 'orange']