import argparse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI # for calling the OpenAI API
import tiktoken  # for counting tokens
import os # for getting API token from env variable OPENAI_API_KEY

import os

class GPTQueryGen:
    def __init__(self, db_name: str, model, token_budget: int = 4096):
        self.model = model
        self.token_budget = token_budget
        self.db_name = db_name 
        self.db = self.load_faiss()
        self.client = self.load_openai_client()

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))

    def query_message(self, query: str, token_budget: int) -> str:
        """Return a message for GPT, with relevant source texts queried from local FAISS db."""
        strings = self.query_faiss(query)
        introduction = 'Use the below website pages from PartSelect.com to answer the subsequent question. If the answer cannot be found in the articles, try and use the info provided but mention that "I could not find a direct answer."'
        question = f"\n\nQuestion: {query}"
        message = introduction

        # Add each string to the message until the token budget is reached. Alwats uses at least one string.
        tokens_used = 0
        tokens_used += self.num_tokens(message + question)
        print(f"Number of Results: {len(strings)}")
        for string in strings:
            next_article = f'\n\PartSelect Webpage:\n"""\n{string}\n"""'
            message += next_article
            tokens = self.num_tokens(next_article)
            tokens_used += tokens
            print(f"Article token length: {tokens}")
            if (tokens_used > token_budget):
                break
        print(f"Tokens Used: {tokens_used}")
        return message + question

    def load_faiss(self):
        embeddings = OpenAIEmbeddings()
        db = FAISS.load_local(self.db_name, embeddings, allow_dangerous_deserialization=True)
        return db 

    def query_faiss(self, query: str) -> list:
        docs = self.db.similarity_search_with_score(query)
        return docs

    def ask(self,
        query: str,
        print_message: bool = False,    
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = self.query_message(query, self.token_budget)
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about applieance parts"},
            {"role": "user", "content": message},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        response_message = response.choices[0].message.content
        return response_message

    def load_openai_client(self):
        """Load the OpenAI client from the API key in the environment."""
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client

def main():
    parser = argparse.ArgumentParser(description="GPT Query Generator CLI")
    parser.add_argument("--db-name", type=str, help="Name of the database")
    parser.add_argument("--query", type=str, help="Query to ask GPT")
    parser.add_argument("--model", type=str, help="GPT model to use", default="gpt-3.5-turbo")
    args = parser.parse_args()

    if not args.db_name:
        print("Please provide a value for --db-name")
        return

    if not args.query:
        print("Please provide a value for --query")
        return

    gpt4 = GPTQueryGen(model=args.model, token_budget=4096, db_name=args.db_name)
    response = gpt4.ask(args.query, print_message=True)
    print(response)

if __name__ == "__main__":
    main()
