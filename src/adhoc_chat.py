import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.system import SystemMessage
from tqdm import tqdm

# Adhoc queries against Claude to find new strategies


def _main():
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
    )

    system_message = SystemMessage(
        """
            You are a senior Financial Analyst working for an Australian financial institution.
            You are given International Financial Market Reports.
            You find salient points in those documents.
            
            Your job is to write clear and accurate instructions.

            The instruction must satisfy the folllowing:
            - The instruction must be 1 sentence long.
            - The instruction must describe a search strategy to extract an insightful fact.
            - The instruction must be applicable to a wide range of financial documents.
            - The instruction must be for a single fact.
            - The instruction must be detailed enough to find non-obvious facts.

            Here is an example of an instruction:
            <example>
            Find a significant fact that a Financial Analyst would find important, where that fact is only apparent when considering two different pages of the document. The fact must not be apparent when looking at just one page.
            </example>

            Output only the search instruction. 
            Do not include any preamble or repeat the question.
        """
    )

    messages = [
        # HumanMessage(
        #     content=f"""
        #         Generate a search instruction within for the type of fact that you'd look for first when reviewing a document.
        #     """
        # ),
        HumanMessage(
            content=f"""
            Generate a search instruction for facts that aren't always apparent by doing a superficial scan of the document.
        """
        ),
    ]

    n = 3
    for message in messages:
        print(f"{20*'='}\n{message.content}\n{20*'='}")

        for i in range(n):
            result = model.invoke(input=[system_message, message]).content

            print(f"{result}")


if __name__ == "__main__":
    _main()

    # Your job is to review Financial Market Analysis Reports and extract the pertinent facts.
# You
# You are an test Examiner for post-graduate Financial Analysts students. You create challenging insightful exam questions. In 1 sentence only, create an exam question that can be applied to a financial market report that will test the students understanding of nuanced financial reasoning.

# From what you understand of Financial Market Reports, generate a single, one-sentence instruction for how to find a certain category of information in these documents.
# There is usually a lot of background and numeric backing-up of facts. Ensure your instruction asks the user to look for a category of fact in a generic, non-document-specific way

# You are a postgraduate Financial Analyst from a top university. You have access to the latest financial models and thinking. You consider trending research in your field when you give an answer. You have been given a 50 page financial market report and you need to find 5 of the most pertinent facts from the document. How would you give an undergraduate student instructions on how to find those 5 facts. List those 5 instructions in such a way that they are unambiguous but also detailed and effective.
