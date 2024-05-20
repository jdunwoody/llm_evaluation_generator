import csv
import re
from pathlib import Path
from pprint import pprint

import boto3
import pandas as pd
import xmltodict
from langchain_aws import ChatBedrock
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from tqdm import tqdm


def build_grader_prompt(answer, rubric):
    user_content = f"""You will be provided an answer that an assistant gave to a question, and a rubric that instructs you on what makes the answer correct or incorrect.
    
    Here is the answer that the assistant gave to the question.
    <answer>{answer}</answer>
    
    Here is the rubric on what makes the answer correct or incorrect.
    <rubric>{rubric}</rubric>
    
    An answer is correct if it entirely meets the rubric criteria, and is otherwise incorrect. =
    First, think through whether the answer is correct or incorrect based on the rubric inside <thinking></thinking> tags. Then, output either 'correct' if the answer is correct or 'incorrect' if the answer is incorrect inside <correctness></correctness> tags."""

    messages = [{"role": "user", "content": user_content}]
    return messages

    # You will be given an instruction that you must follow precisely.

    # <example>
    #     Find a contradictory statement within the first 3 pages of the document that isn't backed-up later in the document
    # </example>


def answer_message(i, instruction, text) -> HumanMessage:
    message = HumanMessage(
        content=f"""{i}.
        You will be provided an instruction for you to follow and some text to analyse. 
        You will generate an answer that follows the instruction for this text and the page numbers(s) within the text that you found it.

        Here is an example:

        <example>
            <instruction>Find an obscure economic observation that is unique to this document.</instruction>
            <text>The Ninth Circuit overruled Berkeley’s natural gas ban in new buildings after concluding that it conflicts with Federal law; the impact of this decision could be material.  Natural gas is cheaper than electricity per unit of energy, offsetting heat pump efficiency benefits for homeowners</text>
            <page>2</page>

            <answer>Natural gas is a better choice for US states, even if they have regulation that opposes it.</answer>
        </example>

        Given this example, here is a user generated instruction and text to generate an answer for:

        <instruction>{instruction}</instruction>
        <text>{text}</text>

        Based on the guidelines above, generate an answer within <answer></answer> tags and page within <page></page> tags. Include only the actual answer, do not include the instruction or any preamble within the question.
    """
    )

    return message


def question_message(i, text, answer) -> HumanMessage:
    message = HumanMessage(
        content=f"""{i}
        You will be given an answer and text.
        You will generate an question that would naturally lead to this answer for the supplied text.

        Here is an example:
        <example>
            <text>if all EV charging were evenly distributed throughout the day, incremental capacity needs would only be 1 GW and peak loads would only rise by 17%. </text>
            <answer>Natural gas is a better choice for US states, even if they have regulation that opposes it.</answer>
                
            <question>If legal protection is in place, should a state still push for a particular energy generation approach?</question>
        </example>

        Given this example, here is a user generated instruction text and answer to generate a question for:

        <text>{text}</text>
        <answer>{answer}</answer>

        Based on the guidelines above, generate a question within <question></question> tags. Include only the actual question, do not include the instruction, answer or any preamble within the question.
    """
    )

    return message


def generate_question(i, model, system_message, answer, text):
    messages = [system_message, question_message(i=i, text=text, answer=answer)]
    question_xml = model.invoke(input=messages).content
    # question = xmltodict.parse(question_xml)["question"]
    try:
        question = re.findall(r"<question>(.*)</question>", question_xml)[0]
    except:
        question = question_xml

    return question


def generate_answer(i, model, system_message, instruction, text):
    messages = [system_message, answer_message(i=i, instruction=instruction, text=text)]
    answer_xml = model.invoke(input=messages).content
    # answer_dict = xmltodict.parse(answer_xml)
    try:
        answer = re.findall(r"<answer>(.*)</answer>", answer_xml)[0]
    except:
        answer = answer_xml

    try:
        page = re.findall(r"<page>(.*)</page>", answer_xml)[0]
    except:
        page = answer_xml

    return answer, page


def _main():
    cache_path = Path(__file__).parent / ".langchain.db"
    # if cache_path.exists():
    #     cache_path.unlink()

    set_llm_cache(SQLiteCache(database_path=cache_path))
    cache_llm = True
    n = 1

    text = load_doc()

    evaluation_category_prompts = {
        "single_true_fact": "Find a single significant fact that a Financial Analyst would find significant, that only appears once in the document.",
        "single_false_fact": "Make up a single fact that a Financial Analysit would normally be looking for, but it definitely must not appear in the document.",
        "split_true_fact": "Find a significant fact that a Financial Analyst would find important, where that fact is only apparent when considering two different pages of the document. The fact must not be apparent when looking at just one page.",
        "key_metric": "Identify a key financial metric or trend by comparing data points across different sections of the document.  ",
        "contradiction": "Search for a statistic or data point that contradicts a commonly held assumption or conventional wisdom within the financial industry.  ",
        "evolving_fact": "Identify a key financial metric or trend that is mentioned across multiple sections of the document, and analyze how it evolves or is contextualized differently in each section.  ",
        "executive_summary": "Carefully read the Executive Summary as it provides a snapshot of the entire report.  Note down any highlighted trends, major findings, and significant conclusions presented in this section.",
        "economic_indicators": "Locate the section discussing economic indicators (e.g., GDP growth rates, inflation, employment data).  Write down the current values and any changes compared to previous periods, as well as any analysis provided on how these indicators are impacting the market.",
        "sector_performance": "Focus on the Sector Analysis section to determine how different sectors are performing.  Identify the best-performing and worst-performing sectors, noting any specific reasons or factors mentioned for their performance.",
        "forward_looking_statements": "Pay attention to the Future Outlook or Projections sections towards the end of the report.  Extract key forecasts, expectations, and any strategic recommendations or anticipated challenges that could influence future market conditions.",
        "geopolitical_factors": "Identify any geopolitical events or conditions discussed in the document that are influencing the market, and summarize their potential impact.",
        "investment_sentiment": "Assess and summarize the overall market sentiment towards investment, including any shifts in investor confidence or behavior highlighted in the report.",
        "risk_factor": "Identify a key risk factor or economic trend that could significantly impact the financial institution's performance or outlook, by carefully examining sections discussing macroeconomic conditions, regulatory changes, and industry dynamics.",
        "cross_reference": "Carefully cross-reference data points across multiple sections of the document to uncover non-obvious trends, discrepancies or interconnected insights that may not be evident from a cursory review.",
        "key_metrics_time_periods": "Identify key financial metrics or trends by carefully comparing data across multiple time periods within the document.",
        "counterintuitive": "Find a counterintuitive or unexpected fact that contradicts conventional wisdom by carefully analyzing data and trends across multiple sections of the financial report.",
    }

    results = []

    system_message = SystemMessage(
        """
            You are a financial analyst who deeply reviews Financial Market Reports.
        """,
    )
    model = load_model(cache_llm=cache_llm)

    progress = tqdm("Generating examples", total=len(evaluation_category_prompts) * n)

    for category, instruction in evaluation_category_prompts.items():
        for i in range(n):
            progress.set_description_str(f"{i+1}/{n}:{category}")

            answer, page = generate_answer(
                i=i,
                model=model,
                system_message=system_message,
                text=text,
                instruction=instruction,
            )
            question = generate_question(
                i=i,
                model=model,
                system_message=system_message,
                text=text,
                answer=answer,
            )

            result = {
                "i": i,
                "category": category,
                "instruction": instruction,
                "question": question,
                "page": page,
                "answer": answer,
            }

            results.append(result)
            print(question)
            print(answer)

            progress.update(1)

    output_path = Path(__file__).parents[1] / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data=results)
    df.to_csv(
        output_path / "eval.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        header=True,
    )
    df.to_json(output_path / "eval.json", default_handler=str, orient="records")


def load_doc() -> str:
    output_dir = Path(__file__).parents[1] / "output"
    doc_dir = output_dir / "JPM Electravision 14th Annual Energy Paper 20240305.pdf.txt"

    return doc_dir.read_text()


def load_model(cache_llm):
    model = ChatBedrock(
        client=boto3.client(
            service_name="bedrock-runtime",
            region_name="ap-southeast-2",
        ),
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        model_kwargs={
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_k": 250,
            "top_p": 1,
            "stop_sequences": ["\n\nHuman"],
        },
        cache=cache_llm,
    )

    return model


if __name__ == "__main__":
    _main()

one_shot_text = """
        The second explanation has to do with the chemistry of industrial energy use, which Lawrence Berkeley National Laboratory examined in a piece on high and low potential for electrification.  LBNL cited primary metals (ex-steel), secondary steel 10 , machinery, wood products, plastics and rubber as sectors with high electrification potential since fossil fuels are mostly used for process heat which could be replaced by electric heat.  Electrification potential is also high for certain mining activities related to transport, excavation, pit crushing and belt conveying systems.  That’s the table on the left.  On the right: low/medium electrification potential sectors .  Chemicals, pulp/paper and food take advantage of integrated systems in which fuel combustion waste heat (CHP) powers related processes.  CHP-intensive sectors are harder to electrify since producers would need to purchase energy previously obtained at little to no cost, and/or redesign the entire process.  Other hard to electrify sectors include non-metallic minerals such as glass, brick and cement which require temperatures in excess of 1400°C, and which are non-conductive solids (i.e., harder to electrify production of things that do not conduct electricity).  Finally, oil/coal refining exploits “own- use” fuel consumption, a source of energy lost when switching to electricity.
"""

one_shot_analysis = """
'1. Identify the main topic: The text seems to be discussing the potential '
 'for electrification in various industrial sectors and the factors that '
 'influence this potential.\n'
 '\n'
 '2. Understand the context: The text mentions the Lawrence Berkeley National '
 'Laboratory (LBNL) and their analysis of the chemistry of industrial energy '
 'use, which forms the basis for the discussion.\n'
 '\n'
 '3. Categorize the sectors:\n'
 '   a. High electrification potential sectors:\n'
 '      - Primary metals (e.g., steel)\n'
 '      - Secondary steel\n'
 '      - Machinery\n'
 '      - Wood products\n'
 '      - Plastics and rubber\n'
 '      - Certain mining activities (transport, excavation, pit crushing, belt '
 'conveying systems)\n'
 '   b. Low/medium electrification potential sectors:\n'
 '      - Chemicals\n'
 '      - Pulp/paper\n'
 '      - Food\n'
 '\n'
 '4. Identify the reasons for high electrification potential:\n'
 '   - Fossil fuels are mainly used for process heat, which could be replaced '
 'by electric heat.\n'
 '\n'
 '5. Identify the reasons for low/medium electrification potential:\n'
 '   a. Sectors like chemicals, pulp/paper, and food take advantage of '
 'integrated systems where fuel combustion waste heat (combined heat and '
 'power, CHP) powers related processes.\n'
 '   b. CHP-intensive sectors are harder to electrify because:\n'
 '      - Producers would need to purchase energy previously obtained at '
 'little to no cost.\n'
 '      - Entire processes might need to be redesigned.\n'
 '   c. Non-metallic minerals (e.g., glass, brick, cement) require very high '
 'temperatures (over 1400°C) and are non-conductive solids, making '
 'electrification more challenging.\n'
 '   d. Oil/coal refining exploits "own-use" fuel consumption, which would be '
 'lost when switching to electricity.\n'
 '\n'
 '6. Summarize the key points:\n'
 '   - The text categorizes industrial sectors based on their potential for '
 'electrification, with some sectors having high potential due to the use of '
 'fossil fuels for process heat, while others have low/medium potential due to '
 'factors like integrated systems, high temperature requirements, and the use '
 'of "own-use" fuel consumption.\n'
 '   - The analysis highlights the challenges and considerations involved in '
 'electrifying different industrial sectors, which can vary based on their '
 'specific processes and energy requirements.\n'
"""

# When you are given text like: {one_shot_text}

# You analyse it in the following way:
# {one_shot_analysis}
# you generate JSON with just question and answer keys:
# {
#     "question":"What integrated systems make use of waste heat?"
#     "reference":"Chemicals, pulp/paper and food take advantage of integrated systems in which fuel combustion waste heat (CHP) powers related processes."
# }
# question = "What integrated systems make use of waste heat?"
# answer:
# question:{question}
# Deeply analyse the following text and make sure you summarising the important points without skipping important parts:
# {text}
# Think deeply about the following text and find a single "question" and "reference" that are uniquely defined in the text:

# Generate an ANSWER that satisfies the INSTRUCTION
# 3. Thinking about the ANSWER and generate a QUESTION
# 4. You remember where the ANSWER was found in the TEXT

# You always respond in the following format:

# QUESTION:
# ANSWER:
# REFERENCE:

# Here is an example:


# QUESTION: What would be the cost to the US in the next few decades?

# ANSWER: The monetary costs to the US would be $300 billion by 2040. This was found on page 4 of the text.
# Given the following INSTRUCTION: {prompt_body}

# and the following TEXT: {text}

# respond with an accurate and succinct QUESTION along with a single succinct ANSWER
