from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
import configparser
def generate_summary(problem, query, method, function_out, n_sentences):
    config = configparser.ConfigParser()
    config.read('app.config')
    groq_api_key = config.get('API_KEYS', 'groq_api_key')
    model_name = config.get('MODEL_SETTINGS', 'model_name')
    temperature = config.getfloat('MODEL_SETTINGS', 'temperature')

    CYPHER_QA_TEMPLATE = """(A) is a list of information that includes i) the original causal
problem, ii) the class identification of the causal problem, iii)
the used method, and iv) the outcomes.
Interpret the results in (A) in response to the original causal problem,
using neutral language to paraphrase it more fluently and engagingly.
The output summary is (I)
please write a simple and clear summary.
Summary should be easy to understand for someone who does not have a background in statistics, focusing on the practical meaning and implications of result.
Explain in summary what the numbers mean in simple terms, why they are important, and how they might affect decisions in a practical context.
Guidelines:
1: (I) must concentrate on interpreting the result provided in (A)
in response to the problem.
2: (I) must include all the results, methods, and dataset name in (A).
3: (I) may include jargon from (A), but it should not include any
other technical terms
not mentioned in (A).
4: The problem in (A) is a causal problem, thus (I) should not
interpret the results as correlation or association.
5: (I) should use a diversified sentence structure that is also
reader-friendly and concise, rather than listing information one by one.
6: Instead of including the problems, (I) should use the original
problem to develop a more informative interpretation of the result.
7: (I) has to avoid using strong qualifiers such as ’significant’.
8: (I) has to be {n_sentences} sentences or less long, with no
repetition of contents.
9: (I) must not comment on the results.
(A):
i) original causal problem: {query}
ii) class identification of the causal problem: {problem}
iii) used method: {method}
iv) outcomes: {function_out}
(I):

"""

    # Create the prompt template
    qa_prompt = ChatPromptTemplate.from_template(CYPHER_QA_TEMPLATE)
    output_parser = StrOutputParser()
    
    # Initialize the model
    llm = ChatGroq(temperature=temperature, model_name=model_name, groq_api_key=groq_api_key)

    # Create the chain
    chain = qa_prompt | llm | output_parser
    
    # Generate the summary
    result = chain.invoke({"problem": problem, "query": query, "method": method, "function_out": function_out, "n_sentences": n_sentences})
    
    return result