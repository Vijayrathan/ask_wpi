# Requirements: pip install pdfplumber
import os
import json
import pdfplumber



SITE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'site_data')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'dataset.jsonl')

GENERIC_QUESTION = "What is this document about?"
MAX_ANSWER_CHARS = 1000 




def create_dataset():
    pdf_files = [f for f in os.listdir(SITE_DATA_DIR) if f.lower().endswith('.pdf')]

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(SITE_DATA_DIR, pdf_file)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text = "\n".join(page.extract_text() or '' for page in pdf.pages)
                text = text.strip()
                if text:
                    answer = text[:MAX_ANSWER_CHARS].strip()
                    qa_obj = {
                        "messages": [
                            {"role": "user", "content": GENERIC_QUESTION},
                            {"role": "assistant", "content": answer}
                        ]
                    }
                    json.dump(qa_obj, out_f, ensure_ascii=False)
                    out_f.write('\n')
                    print(f"Processed: {pdf_file}")
                else:
                    print(f"Warning: No text extracted from {pdf_file}")
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
