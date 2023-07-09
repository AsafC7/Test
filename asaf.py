import ast
import openai
import os
import pandas as pd
from PyPDF2 import PdfReader
import re
import tiktoken

def extract_text_from_pdf(file_path):
    pdf = PdfReader(file_path)
    text_pages = []
    for page in pdf.pages:
        extracted_page_text = page.extract_text()
        cleaned_page_text = re.sub(r'^\s*\d+\.?\s*', '', extracted_page_text, flags=re.MULTILINE)
        text_pages.append(cleaned_page_text)
    text = "".join(text_pages)
    return text

def extract_text(input_text):
    start_patterns_primary = [r"FIRST CLAIM FOR RELIEF", r"FIRST CAUSE OF ACTION", r"FIRST COA", r"FIRST CAUSE", r"COUNT 1", r"COUNT ONE", r"COUNT I"]
    start_patterns_secondary = [r"CLASS ALLEGATIONS", r"COLLECTIVE ACTION ALLEGATIONS", r"CLASS ACTION ALLEGATIONS"]
    end_pattern_primary = [r"PRAYER FOR RELIEF", r"RELIEF SOUGHT", r"RELIEF REQUESTED", r"prayer"]
    end_pattern_secondary = [r"Respectfully submitted", r"TRIAL BY JURY IS DEMANDED", r"DEMAND FOR JURY TRIAL"]

    start_match = None
    for pattern in start_patterns_primary + start_patterns_secondary:
        start_match = re.search(pattern, input_text)
        if start_match is not None:
            break

    if start_match is None:
        print("No start pattern found in text.")
        return

    end_match = None
    for pattern in end_pattern_primary + end_pattern_secondary:
        end_match = re.search(pattern, input_text)
        if end_match is not None:
            break

    if end_match is None:
        print("No end pattern found in text. Extracting until end of the text.")
        return input_text[start_match.start():]

    return input_text[start_match.start():end_match.start()]

def remove_last_sentence(text):
    # Find the last occurrence of a period ('.') followed by a space (' ')
    last_period_index = text.rfind('. ')
    
    if last_period_index != -1:
        # Remove the last sentence by excluding it from the text
        text = text[:last_period_index+1]
    
    return text

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def process_text_with_chat_gpt(case, prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role":"user","content":prompt + case}],
        temperature=0
    )
    return completion.choices[0].message['content']

def main():
    MAX_TOKENS = 16000
    PROMPT = """
       As an experienced class action lawyer, your task is to critically analyze individual class action complaint texts and identify the distinct legal causes of action (CoAs). Explicit CoAs often follow indications like "FIRST CAUSE OF ACTION", "SECOND CAUSE OF ACTION", and so on. Construct a systematic "Tree of Thoughts" by determining all unique CoAs within each complaint. Provide educated guesses for their general names and the corresponding U.S. law violations and their sections if they are not directly stated. Organize this data into a list of list, each contained list for a CoA, such as the first value of the contained list is the general name of the CoA (drop the words "violation of" and write only the general name of the CoA), and the second value is the full name including the act & sections (if included in the complaint). maintaine the entries in the same order they occur in the complaint. Disregard any potential errors or inconsistencies in the referenced laws or their sections.
       Example output:
       [["Unfair Competition Law", "Cal. Bus. & Prof. Code §§ 17200, et seq."],
       ["Fraud", ""],]
        """
    directory = r"C:\Users\asaf.cohen\Downloads\complaints"
    count = 1
    total_tokens = 0
    df = pd.DataFrame(columns=["name", "general CoA name", "full CoA name"])
    for filename in os.listdir(directory):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            full_text = extract_text_from_pdf(file_path)
            extracted_text = extract_text(full_text)
            if extracted_text:
                num_tokens = num_tokens_from_string(PROMPT, "cl100k_base") + num_tokens_from_string(extracted_text, "cl100k_base")
                print(f"{count}. {filename}: {num_tokens} tokens")
                while num_tokens > MAX_TOKENS:
                    extracted_text = remove_last_sentence(extracted_text)
                    num_tokens = num_tokens_from_string(PROMPT, "cl100k_base") + num_tokens_from_string(extracted_text, "cl100k_base")
                gpt_output = process_text_with_chat_gpt(extracted_text, PROMPT)
                if isinstance(gpt_output, str):
                    try:
                        gpt_output = ast.literal_eval(gpt_output)
                    except ValueError:
                        print(f"Error: Could not parse gpt_output: {gpt_output}")
                        return  # or handle this error however you wish
                # Check that gpt_output is a list of lists and each sublist contains at least two strings
                if not all(isinstance(output, (list, tuple)) and len(output) >= 2 and all(isinstance(elem, str) for elem in output) for output in gpt_output):
                    print(f"Warning: Unexpected gpt_output: {gpt_output}")
                else:
                    updated_gpt_output = list(map(lambda output: [output[0].lower().replace("violation of the ", "").replace("violations of the ", "").replace("violation of ", "").replace("violations of ", "")] + output[1:], gpt_output))
                    print(f"GPT output:\n{updated_gpt_output}")
                    # Add the updated data to the DataFrame
                    for output in updated_gpt_output:
                        new_row = {"name": filename, "general CoA name": output[0], "full CoA name": output[1]}
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                total_tokens += num_tokens
                count += 1
    average_tokens = total_tokens / count if count > 0 else 0
    print(f"\nAverage tokens per file: {average_tokens}")
    return df

if __name__ == "__main__":
    df = main()
    print(df)
