import argparse
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI 
import tiktoken  
import os 
import re 
import os
import time
import string

class GPTQueryGen:
    def __init__(self, model, embeddings: str, db_name=None, db_instance=None, token_budget: int = 4096):
        self.model = model
        self.token_budget = token_budget
        self.embeddings = embeddings
        if db_instance is None:
            self.db = self.load_faiss(db_name)
        else:
            self.db = db_instance
        self.client = self.load_openai_client()
        self.previous_answers = []
        self.previous_db_results = []

    def num_tokens(self, text: str) -> int:
        """Return the number of tokens in a string."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))

    def query_message(self, query: str, token_budget: int) -> str:
        """Return a message for GPT, with relevant source texts queried from local FAISS db."""        
        strings = self.query_faiss(query)
        introduction = 'Use the below website pages from PartSelect.com and previous answers you have given to answer the subsequent question. ALWAYS link to relevant sources. If the answer cannot be found in the articles, try and use the info provided but mention that "I could not find a direct answer."'
        question = f"\n\nQuestion: {query}"
        message = introduction
        
        # Add each string to the message until the token budget is reached. Alwats uses at least one string.
        tokens_used = 0
        tokens_used += self.num_tokens(message + question)
        previous_answer_budget = 1024
        previous_db_output_budget = 1024
        print(f"Number of Results: {len(strings)}")
        if len(strings) == 0:
            return None
        for string in strings:
            next_article = f'\n\PartSelect Webpage:\n"""\n{string}\n"""'
            message += next_article
            tokens = self.num_tokens(next_article)
            tokens_used += tokens
            print(f"Article token length: {tokens}")
            if (tokens_used > token_budget-previous_answer_budget-previous_db_output_budget):
                break
        print(f"Tokens Used: {tokens_used}")

        # use 1 previous article from every previous query until run out of tokens or articles
        for i in range(len(self.previous_db_results)-1, -1, -1): # iterate in reverse
            docs = self.previous_db_results[i]
            if docs:
                 doc = docs[0:2] 
            else: 
                continue
            for d in doc:
                next_article = f'\n\PartSelect Webpage:\n"""\n{d}\n"""'
                message += next_article
                tokens = self.num_tokens(next_article)
                tokens_used += tokens
                print(f"Previous Article token length: {tokens}")
                if (tokens_used > token_budget-previous_answer_budget):
                    break

        # add previous GPT answers to the message
        
        for i in range(len(self.previous_answers)-1, -1, -1): # iterate in reverse
            answer = self.previous_answers[i]
            message += f'\n\Previous Answer:\n"""\n{answer}\n"""'
            tokens = self.num_tokens(answer)
            print(f"Prev ansswer token length: {tokens}")
            tokens_used += tokens
            if (tokens_used > token_budget):
                break

        return message + question

    def load_faiss(self, db_name):
        embeddings = None
        if self.embeddings == "hf":
            embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            model_kwargs = {'device': 'mps'}
            encode_kwargs = {'batch_size': 8}
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        elif self.embeddings == "openai":
            embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        db = FAISS.load_local(db_name, embeddings, allow_dangerous_deserialization=True)
        return db 

    def query_faiss(self, query: str) -> list:
        ps_num = self.find_part_select_num(query)
        model_num = self.find_model_num(query)
        if model_num == ps_num:
             model_num = "N/A" 

        print(f"Model Number: {model_num}")
        print(f"Part Select Number: {ps_num}")

        docs = None
        if model_num != "N/A" and ps_num != "N/A":
            docs = self.db.similarity_search_with_score(query, 
                filter=lambda d: d["model_num"]== model_num and d["ps_num"] == ps_num, 
                k=25, fetch_k=60000)
        elif model_num != "N/A":
            print("Model Number Search")
            docs = self.db.similarity_search_with_score(query, 
                filter=lambda d: d["model_num"]== model_num, 
                k=25, fetch_k=60000)        
        elif ps_num != "N/A":
            print("Part Number Search")
            docs = self.db.similarity_search_with_score(query, 
                filter=lambda d: d["ps_num"]== ps_num, 
                k=20, fetch_k=60000)
        else: 
            docs = self.db.similarity_search_with_score(query, k=20)
        self.previous_db_results.append(docs)
        return docs
    
    def find_model_num(self, title):
        max_number_count = 3 # hard code to 3 to avoid false positives
        part_num = "N/A"
        
        # Split the title into words and iterate over them
        for word in title.split():
            # Strip punctuation from the beginning and end of the word
            word = word.strip(string.punctuation)

            # Count the number of digits in the word
            number_count = sum(1 for char in word if char.isdigit())
            if number_count > max_number_count:
                max_number_count = number_count
                part_num = word
        return part_num
    
    def find_part_select_num(self, url):
        # Regular expression to match "PS" followed by digits
        pattern = r'PS\d+'
        match = re.search(pattern, url)
        if match:
            return match.group(0)  # Return the matched part select number
        else:
            return "N/A"
    def ask(self,
        query: str,
        print_message: bool = False,    
    ) -> str:
        """Answers a query using GPT and a dataframe of relevant texts and embeddings."""
        message = self.query_message(query, self.token_budget)
        if message is None:
            return "I could not find a direct answer."
        if print_message:
            print(message)
        messages = [
            {"role": "system", "content": "You answer questions about appliances."},
            {"role": "user", "content": message},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        response_message = response.choices[0].message.content
        self.previous_answers.append(response_message)
        self.previous_db_results.append(query)
        return response_message

    def load_openai_client(self):
        """Load the OpenAI client from the API key in the environment."""
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client

def main():
    parser = argparse.ArgumentParser(description="GPT Query Generator CLI")
    parser.add_argument("--db-name", type=str, help="Name of the database")
    parser.add_argument("--embeddings-model", type=str, help="Embeddings model to use")
    parser.add_argument("--model", type=str, help="GPT model to use", default="gpt-3.5-turbo")
    args = parser.parse_args()

    if not args.db_name:
        print("Please provide a value for --db-name")
        return
    
    if not args.embeddings_model:
        print("Please provide a value for --embeddings-model")
        return

    print("Loading...")
    gpt = GPTQueryGen(model=args.model, embeddings=args.embeddings_model, token_budget=4096, db_name=args.db_name)
    while True:
        query = input("Enter your query: ")
        start = time.time()
        response = gpt.ask(query, print_message=True)
        print(response)
        print(f"Time: {time.time() - start}")

if __name__ == "__main__":
    main()
